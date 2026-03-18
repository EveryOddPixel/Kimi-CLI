/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { NvidiaContentGenerator } from './nvidiaContentGenerator.js';
import { type ContentGeneratorConfig } from './contentGenerator.js';
import { LlmRole } from '../telemetry/llmRole.js';

const mockConfig: ContentGeneratorConfig = {
  apiKey: 'test-key',
  baseUrl: 'https://test.nvidia.com/v1',
  authType: 'nvidia' as any,
};

const server = setupServer();

describe('NvidiaContentGenerator', () => {
  beforeEach(() => server.listen());
  afterEach(() => {
    server.close();
    server.resetHandlers();
  });

  it('should generate content correctly', async () => {
    server.use(
      http.post('https://test.nvidia.com/v1/chat/completions', async ({ request }) => {
        const body = (await request.json()) as any;
        expect(body.model).toBe('test-model');
        expect(body.messages).toHaveLength(1);
        expect(body.messages[0].role).toBe('user');
        expect(body.messages[0].content).toBe('hello');

        return HttpResponse.json({
          choices: [
            {
              message: {
                content: 'hi there',
                reasoning_content: 'thinking...',
              },
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        });
      }),
    );

    const generator = new NvidiaContentGenerator(mockConfig);
    const response = await generator.generateContent(
      {
        model: 'test-model',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
      },
      'prompt-id',
      LlmRole.MAIN,
    );

    expect(response.candidates?.[0]?.content?.parts).toContainEqual({ text: 'hi there' });
    expect(response.candidates?.[0]?.content?.parts).toContainEqual({ thought: 'thinking...' });
    expect(response.usageMetadata?.totalTokenCount).toBe(15);
  });

  it('should handle streaming content correctly', async () => {
    server.use(
      http.post('https://test.nvidia.com/v1/chat/completions', () => {
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue(
              encoder.encode(
                'data: ' +
                  JSON.stringify({
                    choices: [
                      {
                        delta: { reasoning_content: 'thinking' },
                        finish_reason: null,
                      },
                    ],
                  }) +
                  '\n\n',
              ),
            );
            controller.enqueue(
              encoder.encode(
                'data: ' +
                  JSON.stringify({
                    choices: [
                      {
                        delta: { content: 'hello' },
                        finish_reason: 'stop',
                      },
                    ],
                  }) +
                  '\n\n',
              ),
            );
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
          },
        });
        return new HttpResponse(stream, {
          headers: { 'Content-Type': 'text/event-stream' },
        });
      }),
    );

    const generator = new NvidiaContentGenerator(mockConfig);
    const stream = await generator.generateContentStream(
      {
        model: 'test-model',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
      },
      'prompt-id',
      LlmRole.MAIN,
    );

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);
    expect(chunks[0].candidates?.[0]?.content?.parts).toContainEqual({ thought: 'thinking' });
    expect(chunks[1].candidates?.[0]?.content?.parts).toContainEqual({ text: 'hello' });
  });

  it('should throw error on API failure', async () => {
    server.use(
      http.post('https://test.nvidia.com/v1/chat/completions', () => {
        return new HttpResponse('Unauthorized', { status: 401 });
      }),
    );

    const generator = new NvidiaContentGenerator(mockConfig);
    await expect(
      generator.generateContent(
        { model: 'test-model', contents: [] },
        'id',
        LlmRole.MAIN,
      ),
    ).rejects.toThrow('NVIDIA API error: 401 Unauthorized');
  });
});
