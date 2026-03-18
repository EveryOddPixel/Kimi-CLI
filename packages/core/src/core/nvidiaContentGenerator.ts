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
          tool_call_id: p.functionResponse.name, // This is a hack, Gemini format doesn't have IDs
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
    
    debugLogger.debug(`[NVIDIA] Requesting model: ${request.model}`);
    debugLogger.debug(`[NVIDIA] URL: ${targetUrl}`);
    
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
    
    debugLogger.debug(`[NVIDIA] Requesting model: ${request.model}`);
    debugLogger.debug(`[NVIDIA] URL: ${targetUrl}`);
    
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
                  // Reasoning is always safe to yield immediately
                  if (delta.reasoning_content) {
                    yield self.mapFromOpenAiStreamResponse(json);
                    continue;
                  }

                  if (delta.content) {
                    textBuffer += delta.content;

                    // If we are not in a marker, yield everything UP TO the first '<|'
                    if (!markerActive) {
                      const markerIdx = textBuffer.indexOf('<|');
                      if (markerIdx !== -1) {
                        // Yield text before marker
                        const preMarkerText = textBuffer.substring(0, markerIdx);
                        if (preMarkerText) {
                          const response = self.mapFromOpenAiStreamResponse(json);
                          response.candidates![0].content.parts = [{ text: preMarkerText }];
                          yield response;
                        }
                        // Enter marker mode and keep the rest in buffer
                        textBuffer = textBuffer.substring(markerIdx);
                        markerActive = true;
                      } else {
                        // No marker at all, yield everything and clear buffer
                        yield self.mapFromOpenAiStreamResponse(json);
                        textBuffer = '';
                      }
                    } 
                    
                    // If we ARE in a marker, only yield once it closes
                    if (markerActive) {
                      if (textBuffer.includes('<|tool_calls_section_end|>')) {
                        const { text, toolCalls } = self.parseEmbeddedToolCalls(textBuffer);
                        const response = new GenerateContentResponse();
                        response.candidates = [{
                          content: {
                            role: 'model',
                            parts: [...(text ? [{ text }] : []), ...toolCalls]
                          }
                        }];
                        yield response;
                        textBuffer = '';
                        markerActive = false;
                      }
                      // Otherwise, stay silent and keep buffering
                    }
                  } else if (delta.tool_calls) {
                    yield self.mapFromOpenAiStreamResponse(json);
                  }
                }
              } catch (e) {
                // Ignore partial lines
              }
            }
          }
        }
        
        // Final flush
        if (textBuffer) {
          const { text, toolCalls } = self.parseEmbeddedToolCalls(textBuffer);
          const response = new GenerateContentResponse();
          response.candidates = [{
            content: { role: 'model', parts: [...(text ? [{ text }] : []), ...toolCalls] }
          }];
          yield response;
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

    // Regex to match Kimi's tool call format:
    // <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:ID<|tool_call_argument_begin|>JSON_ARGS<|tool_call_end|><|tool_calls_section_end|>
    const sectionRegex = /<\|tool_calls_section_begin\|>([\s\S]*?)<\|tool_calls_section_end\|>/g;
    const callRegex = /<\|tool_call_begin\|>functions\.(\w+):?\d*<\|tool_call_argument_begin\|>([\s\S]*?)<\|tool_call_end\|>/g;

    let match;
    while ((match = sectionRegex.exec(text)) !== null) {
      const sectionContent = match[1];
      let callMatch;
      while ((callMatch = callRegex.exec(sectionContent)) !== null) {
        try {
          toolCalls.push({
            functionCall: {
              name: callMatch[1],
              args: JSON.parse(callMatch[2].trim()),
            },
          });
        } catch (e) {
          debugLogger.warn(`Failed to parse Kimi tool arguments: ${callMatch[2]}`);
        }
      }
      remainingText = remainingText.replace(match[0], '');
    }

    return { text: remainingText.trim(), toolCalls };
  }

  private mapFromOpenAiResponse(data: any): GenerateContentResponse {
    const out = new GenerateContentResponse();
    const choice = data.choices[0];
    
    let content = choice.message.content || '';
    const { text, toolCalls } = this.parseEmbeddedToolCalls(content);
    
    const parts: any[] = [];
    if (choice.message.reasoning_content) {
      parts.push({ thought: choice.message.reasoning_content });
    }
    if (text) {
      parts.push({ text });
    }
    
    // Add standard tool calls
    if (choice.message.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        parts.push({
          functionCall: {
            name: tc.function.name,
            args: JSON.parse(tc.function.arguments),
          },
        });
      }
    }

    // Add parsed embedded tool calls
    parts.push(...toolCalls);

    if (parts.length === 0) {
      parts.push({ text: '' });
    }

    out.candidates = [
      {
        content: {
          role: 'model',
          parts: parts,
        },
        finishReason: choice.finish_reason,
      },
    ];
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
    if (delta && delta.reasoning_content) {
      parts.push({ thought: delta.reasoning_content });
    }
    
    if (delta && delta.content) {
      const { text, toolCalls } = this.parseEmbeddedToolCalls(delta.content);
      if (text) {
        parts.push({ text });
      }
      parts.push(...toolCalls);
    }
    
    if (delta && delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        if (tc.function) {
          parts.push({
            functionCall: {
              name: tc.function.name,
              args: tc.function.arguments ? JSON.parse(tc.function.arguments) : {},
            },
          });
        }
      }
    }

    out.candidates = [
      {
        content: {
          role: 'model',
          parts: parts.length > 0 ? parts : [{ text: '' }],
        },
        finishReason: choice.finish_reason,
      },
    ];
    
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
