/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  type CountTokensResponse,
  type GenerateContentParameters,
  type CountTokensParameters,
  type EmbedContentResponse,
  type EmbedContentParameters,
  type Content,
} from '@google/genai';
import type { ContentGenerator, ContentGeneratorConfig } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import { debugLogger } from '../utils/debugLogger.js';

export class NvidiaContentGenerator implements ContentGenerator {
  userTier?: UserTierId;
  userTierName?: string;
  paidTier?: GeminiUserTier;

  constructor(private readonly config: ContentGeneratorConfig) {}

  private mapToOpenAiMessages(request: GenerateContentParameters): any[] {
    const messages: any[] = [];
    
    if (request.config?.systemInstruction) {
      messages.push({
        role: 'system',
        content: this.mapContentToText(request.config.systemInstruction as any),
      });
    }

    const contents = Array.isArray(request.contents) ? request.contents : [request.contents];
    for (const content of contents as any[]) {
      const role = content.role === 'model' ? 'assistant' : 'user';
      
      const toolCalls = content.parts
        .filter((p: any) => p.functionCall)
        .map((p: any) => ({
          id: `call_${Math.random().toString(36).substring(2, 9)}`,
          type: 'function',
          function: {
            name: p.functionCall.name,
            arguments: JSON.stringify(p.functionCall.args),
          },
        }));

      const functionResponses = content.parts
        .filter((p: any) => p.functionResponse)
        .map((p: any) => ({
          role: 'tool',
          tool_call_id: p.functionResponse.name,
          content: JSON.stringify(p.functionResponse.response),
        }));

      const textContent = this.mapContentToText(content);

      if (textContent || toolCalls.length > 0) {
        messages.push({
          role,
          content: textContent || null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });
      }

      if (functionResponses.length > 0) {
        messages.push(...functionResponses);
      }
    }

    return messages;
  }

  private mapContentToText(content: Content | string | any[]): string {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content.map(p => (typeof p === 'string' ? p : p.text || '')).join('');
    }
    return (content.parts || []).map(p => p.text || '').join('');
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const messages = this.mapToOpenAiMessages(request);
    const body: any = {
      model: request.model,
      messages,
      stream: false,
      temperature: request.config?.temperature ?? 1.0,
      top_p: request.config?.topP ?? 0.9,
      max_tokens: request.config?.maxOutputTokens ?? 16384,
    };

    if (request.tools && request.tools.length > 0) {
      body.tools = request.tools.flatMap((t: any) => 
        t.functionDeclarations?.map((fd: any) => ({
          type: 'function',
          function: {
            name: fd.name,
            description: fd.description,
            parameters: fd.parameters,
          },
        }))
      ).filter(Boolean);
    }

    const baseUrl = this.config.baseUrl?.endsWith('/') 
      ? this.config.baseUrl.slice(0, -1) 
      : (this.config.baseUrl || 'https://integrate.api.nvidia.com/v1');
    const targetUrl = `${baseUrl}/chat/completions`;
    
    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      const error = new Error(`NVIDIA API error: ${response.status} ${errorText}`);
      (error as any).status = response.status;
      throw error;
    }

    const data = (await response.json()) as any;
    return this.mapFromOpenAiResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.mapToOpenAiMessages(request);
    const body: any = {
      model: request.model,
      messages,
      stream: true,
      temperature: request.config?.temperature ?? 1.0,
      top_p: request.config?.topP ?? 0.9,
      max_tokens: request.config?.maxOutputTokens ?? 16384,
    };

    if (request.tools && request.tools.length > 0) {
      body.tools = request.tools.flatMap((t: any) => 
        t.functionDeclarations?.map((fd: any) => ({
          type: 'function',
          function: {
            name: fd.name,
            description: fd.description,
            parameters: fd.parameters,
          },
        }))
      ).filter(Boolean);
    }

    const baseUrl = this.config.baseUrl?.endsWith('/') 
      ? this.config.baseUrl.slice(0, -1) 
      : (this.config.baseUrl || 'https://integrate.api.nvidia.com/v1');
    const targetUrl = `${baseUrl}/chat/completions`;
    
    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      const error = new Error(`NVIDIA API error: ${response.status} ${errorText}`);
      (error as any).status = response.status;
      throw error;
    }

    const reader = (response.body as any).getReader();
    const decoder = new TextDecoder();
    
    const self = this;
    async function* streamGenerator() {
      let buffer = '';
      let textBuffer = '';
      let markerActive = false;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === 'data: [DONE]') continue;
            if (trimmed.startsWith('data: ')) {
              try {
                const json = JSON.parse(trimmed.slice(6));
                const delta = json.choices[0]?.delta;
                
                if (delta) {
                  // 1. Reasoning
                  if (delta.reasoning_content) {
                    const response = new GenerateContentResponse();
                    response.candidates = [{
                      content: { role: 'model', parts: [{ thought: delta.reasoning_content }] }
                    }];
                    yield response;
                  }

                  // 2. Aggressive Scrubber for Content
                  if (delta.content) {
                    textBuffer += delta.content;

                    let continueLoop = true;
                    while (continueLoop && textBuffer.length > 0) {
                      continueLoop = false;
                      
                      if (!markerActive) {
                        const markerIdx = textBuffer.indexOf('<|');
                        if (markerIdx !== -1) {
                          // FOUND MARKER: Flush text BEFORE it
                          const safeText = textBuffer.substring(0, markerIdx);
                          if (safeText) {
                            const response = new GenerateContentResponse();
                            response.candidates = [{ content: { role: 'model', parts: [{ text: safeText }] } }];
                            yield response;
                          }
                          textBuffer = textBuffer.substring(markerIdx);
                          markerActive = true;
                          continueLoop = true;
                        } else {
                          // NO MARKER: Yield text but hold last few chars if they start with '<'
                          let safeLen = textBuffer.length;
                          if (textBuffer.endsWith('<')) safeLen -= 1;
                          else if (textBuffer.endsWith('<|')) safeLen -= 2;
                          
                          if (safeLen > 0) {
                            const safeText = textBuffer.substring(0, safeLen);
                            const response = new GenerateContentResponse();
                            response.candidates = [{ content: { role: 'model', parts: [{ text: safeText }] } }];
                            yield response;
                            textBuffer = textBuffer.substring(safeLen);
                            continueLoop = true;
                          }
                        }
                      } else {
                        // PROTOCOL LOCKED: Check for individual tool_call_end
                        const callEndMarker = '<|tool_call_end|>';
                        const endIdx = textBuffer.indexOf(callEndMarker);
                        if (endIdx !== -1) {
                          const callContent = textBuffer.substring(0, endIdx + callEndMarker.length);
                          const { toolCalls } = self.parseEmbeddedToolCalls(callContent);
                          
                          if (toolCalls.length > 0) {
                            const response = new GenerateContentResponse();
                            response.candidates = [{
                              content: { role: 'model', parts: toolCalls }
                            }];
                            yield response;
                          }
                          
                          textBuffer = textBuffer.substring(endIdx + callEndMarker.length);
                          // We stay in markerActive until the SECTION ends or turn ends
                          if (textBuffer.includes('<|tool_calls_section_end|>')) {
                            textBuffer = textBuffer.replace('<|tool_calls_section_end|>', '');
                            markerActive = false;
                          }
                          continueLoop = true;
                        }
                      }
                    }
                  }

                  // 3. standard tool calls
                  if (delta.tool_calls) {
                    const response = self.mapFromOpenAiStreamResponse(json);
                    response.candidates![0].content.parts = response.candidates![0].content.parts.filter(p => p.functionCall);
                    if (response.candidates![0].content.parts.length > 0) {
                      yield response;
                    }
                  }
                }
              } catch (e) { /* ignore partial */ }
            }
          }
        }
        
        // FINAL CLEANUP: Flush remaining buffer with force-scrub
        if (textBuffer) {
          const { text, toolCalls } = self.parseEmbeddedToolCalls(textBuffer);
          if (text || toolCalls.length > 0) {
            const response = new GenerateContentResponse();
            // Force-scrub any remaining markers from the text part
            const scrubbedText = text.replace(/<\|[\s\S]*?\|>/g, '').trim();
            response.candidates = [{
              content: { role: 'model', parts: [...(scrubbedText ? [{ text: scrubbedText }] : []), ...toolCalls] }
            }];
            yield response;
          }
        }
      } finally {
        reader.releaseLock();
      }
    }

    return streamGenerator();
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    const text = (request.contents as any[]).map(c => this.mapContentToText(c)).join('');
    return { totalTokens: Math.ceil(text.length / 4) };
  }

  async embedContent(_request: EmbedContentParameters): Promise<EmbedContentResponse> {
    throw new Error('Embeddings not supported for NVIDIA provider');
  }

  private parseEmbeddedToolCalls(text: string): { text: string; toolCalls: any[] } {
    const toolCalls: any[] = [];
    let remainingText = text;

    // Resilient Regex: Matches calls with or without the outer section tags
    const callRegex = /<\|tool_call_begin\|>\s*functions\.([\w.]+)(?::\d+)?\s*<\|tool_call_argument_begin\|>([\s\S]*?)<\|tool_call_end\|>/g;

    let match;
    while ((match = callRegex.exec(text)) !== null) {
      try {
        const argsText = match[2].trim();
        toolCalls.push({
          functionCall: {
            name: match[1],
            args: JSON.parse(argsText),
          },
        });
        remainingText = remainingText.replace(match[0], '');
      } catch (e) {
        debugLogger.warn(`[PROTOCOL] Failed to parse tool arguments: ${match[2]}`);
      }
    }

    // Scrub all Kimi markers from the text
    remainingText = remainingText.replace(/<\|[\s\S]*?\|>/g, '').trim();

    return { text: remainingText, toolCalls };
  }

  private mapFromOpenAiResponse(data: any): GenerateContentResponse {
    const out = new GenerateContentResponse();
    const choice = data.choices[0];
    
    const rawContent = choice.message.content || '';
    const { text, toolCalls } = this.parseEmbeddedToolCalls(rawContent);
    
    const parts: any[] = [];
    if (choice.message.reasoning_content) {
      parts.push({ thought: choice.message.reasoning_content });
    }
    if (text) {
      parts.push({ text });
    }
    
    if (choice.message.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        parts.push({
          functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) },
        });
      }
    }

    parts.push(...toolCalls);

    if (parts.length === 0) {parts.push({ text: '' });}

    out.candidates = [{ content: { role: 'model', parts: parts }, finishReason: choice.finish_reason }];
    out.usageMetadata = {
      promptTokenCount: data.usage?.prompt_tokens,
      candidatesTokenCount: data.usage?.completion_tokens,
      totalTokenCount: data.usage?.total_tokens,
    };
    return out;
  }

  private mapFromOpenAiStreamResponse(data: any): GenerateContentResponse {
    const out = new GenerateContentResponse();
    const choice = data.choices[0];
    const delta = choice.delta;
    const parts: any[] = [];
    if (delta && delta.reasoning_content) {parts.push({ thought: delta.reasoning_content });}
    if (delta && delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.function) {
          parts.push({
            functionCall: { name: tc.function.name, args: tc.function.arguments ? JSON.parse(tc.function.arguments) : {} },
          });
        }
      }
    }
    out.candidates = [{ content: { role: 'model', parts: parts.length > 0 ? parts : [{ text: '' }] }, finishReason: choice.finish_reason }];
    if (data.usage) {
      out.usageMetadata = {
        promptTokenCount: data.usage.prompt_tokens,
        candidatesTokenCount: data.usage.completion_tokens,
        totalTokenCount: data.usage.total_tokens,
      };
    }
    return out;
  }
}
