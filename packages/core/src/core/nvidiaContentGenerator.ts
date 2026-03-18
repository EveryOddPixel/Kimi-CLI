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
      if (typeof content === 'string') {
        messages.push({ role: 'user', content });
      } else if (Array.isArray(content)) {
        messages.push({ role: 'user', content: this.mapContentToText(content) });
      } else {
        const role = content.role === 'model' ? 'assistant' : 'user';
        messages.push({
          role,
          content: this.mapContentToText(content),
        });
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
    const body = {
      model: request.model,
      messages,
      stream: false,
      temperature: request.config?.temperature ?? 1.0,
      top_p: request.config?.topP ?? 0.9,
      max_tokens: request.config?.maxOutputTokens ?? 16384,
    };

    const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
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
    const body = {
      model: request.model,
      messages,
      stream: true,
      temperature: request.config?.temperature ?? 1.0,
      top_p: request.config?.topP ?? 0.9,
      max_tokens: request.config?.maxOutputTokens ?? 16384,
    };

    const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
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
                yield self.mapFromOpenAiStreamResponse(json);
              } catch (e) {
                // Ignore partial lines
              }
            }
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

  private mapFromOpenAiResponse(data: any): GenerateContentResponse {
    const out = new GenerateContentResponse();
    const choice = data.choices[0];
    
    const parts: any[] = [];
    if (choice.message.reasoning_content) {
      parts.push({ thought: choice.message.reasoning_content });
    }
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }
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
      parts.push({ text: delta.content });
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
