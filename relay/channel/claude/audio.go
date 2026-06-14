package claude

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/logger"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/types"

	"github.com/gin-gonic/gin"
)

// MiMo 的 TTS/ASR 能力复用 Chat Completions 协议，并不走 Anthropic 的 /v1/messages，
// 也不是 OpenAI 标准的 /v1/audio/* 端点。本文件负责把标准的 OpenAI Audio 请求转换为
// MiMo 所需的 Chat Completions 请求体，并把上游响应转换回客户端期望的格式。

// getTTSFormat 返回 TTS 期望的音频格式，缺省为 pcm（与 MiMo 默认及客户端保持一致）。
func getTTSFormat(request dto.AudioRequest) string {
	if request.ResponseFormat != "" {
		return request.ResponseFormat
	}
	return "pcm"
}

// convertMiMoTTSRequest 把标准 OpenAI /audio/speech 请求转换为 MiMo TTS 的 Chat
// Completions 请求体：messages 同时包含 user（风格指令）与 assistant（朗读文本），
// 音频参数通过顶层 audio 字段指定。
func convertMiMoTTSRequest(request dto.AudioRequest, isStream bool) (io.Reader, error) {
	instruction := request.Instructions
	if instruction == "" {
		instruction = "用自然的语气朗读"
	}
	voice := request.Voice
	if voice == "" {
		voice = "default_zh"
	}

	ttsBody := map[string]any{
		"model": request.Model,
		"messages": []map[string]string{
			{"role": "user", "content": instruction},
			{"role": "assistant", "content": request.Input},
		},
		"max_tokens": 1024,
		"stream":     isStream,
		"audio": map[string]string{
			"voice":  voice,
			"format": getTTSFormat(request),
		},
	}

	jsonBytes, err := common.Marshal(ttsBody)
	if err != nil {
		return nil, fmt.Errorf("marshal mimo tts request: %w", err)
	}
	return bytes.NewReader(jsonBytes), nil
}

// convertMiMoASRRequest 把 multipart 上传的音频转换为 MiMo ASR 的 Chat Completions
// 请求体：content 为单个 input_audio part，音频以 base64(WAV) 内嵌。
func convertMiMoASRRequest(c *gin.Context, request dto.AudioRequest) (io.Reader, error) {
	formData, err := common.ParseMultipartFormReusable(c)
	if err != nil {
		return nil, fmt.Errorf("parse multipart form: %w", err)
	}

	fileHeaders := formData.File["file"]
	if len(fileHeaders) == 0 {
		return nil, errors.New("file is required")
	}

	file, err := fileHeaders[0].Open()
	if err != nil {
		return nil, fmt.Errorf("open audio file: %w", err)
	}
	defer file.Close()

	audioBytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("read audio file: %w", err)
	}

	asrBody := map[string]any{
		"model": request.Model,
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{
						"type": "input_audio",
						"input_audio": map[string]string{
							"data":   base64.StdEncoding.EncodeToString(audioBytes),
							"format": "wav",
						},
					},
				},
			},
		},
		"max_tokens": 512,
	}

	jsonBytes, err := common.Marshal(asrBody)
	if err != nil {
		return nil, fmt.Errorf("marshal mimo asr request: %w", err)
	}
	return bytes.NewReader(jsonBytes), nil
}

// ClaudeTTSHandler 处理 MiMo TTS 响应。
//
// MiMo TTS 复用 Chat Completions 协议：非流式时音频在 choices[0].message.audio.data，
// 流式时在 choices[0].delta.audio.data，均为 base64 编码。非流式直接把解码后的二进制音频
// 返回给客户端；流式则按 OpenAI speech 的 SSE 约定逐块下发 base64 音频（避免把二进制塞进
// SSE 的 data 行而破坏分帧）。
func ClaudeTTSHandler(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) *dto.Usage {
	defer service.CloseResponseBodyGracefully(resp)

	usage := &dto.Usage{}
	usage.PromptTokens = info.GetEstimatePromptTokens()
	usage.TotalTokens = usage.PromptTokens

	audioFormat := "pcm"
	if audioReq, ok := info.Request.(*dto.AudioRequest); ok && audioReq.ResponseFormat != "" {
		audioFormat = audioReq.ResponseFormat
	}

	if info.IsStream {
		helper.StreamScannerHandler(c, resp, info, func(data string, sr *helper.StreamResult) {
			// 最后一条通常携带 usage，使用上游统计覆盖估算值
			if service.SundaySearch(data, "usage") {
				var simpleResponse dto.SimpleResponse
				if err := common.Unmarshal([]byte(data), &simpleResponse); err == nil && simpleResponse.Usage.TotalTokens != 0 {
					usage.PromptTokens = simpleResponse.Usage.InputTokens
					usage.CompletionTokens = simpleResponse.OutputTokens
					usage.TotalTokens = simpleResponse.TotalTokens
				}
			}

			var chunk struct {
				Choices []struct {
					Delta struct {
						Audio struct {
							Data string `json:"data"`
						} `json:"audio"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := common.Unmarshal([]byte(data), &chunk); err != nil {
				return
			}
			if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Audio.Data == "" {
				return
			}
			// base64 音频以 OpenAI speech.audio.delta 事件下发
			if err := helper.ObjectData(c, map[string]string{
				"type":  "speech.audio.delta",
				"audio": chunk.Choices[0].Delta.Audio.Data,
			}); err != nil {
				sr.Error(err)
			}
		})
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		return usage
	}

	// 非流式：上游统计权威，标记为本地计数，避免计费层对（空的）补全文本重新计 token
	common.SetContextKey(c, constant.ContextKeyLocalCountTokens, true)
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.LogError(c, fmt.Sprintf("failed to read TTS response body: %v", err))
		c.Writer.WriteHeader(http.StatusInternalServerError)
		return usage
	}

	var ttsResp struct {
		Choices []struct {
			Message struct {
				Audio struct {
					Data string `json:"data"`
				} `json:"audio"`
			} `json:"message"`
		} `json:"choices"`
		dto.Usage `json:"usage"`
	}
	if err := common.Unmarshal(bodyBytes, &ttsResp); err != nil {
		// 解析失败时原样透传上游响应，便于排查（多半是上游返回的错误 JSON）
		logger.LogError(c, fmt.Sprintf("failed to unmarshal TTS response: %v", err))
		c.Writer.WriteHeaderNow()
		_, _ = c.Writer.Write(bodyBytes)
		return usage
	}

	if ttsResp.TotalTokens != 0 {
		usage.PromptTokens = ttsResp.InputTokens
		usage.CompletionTokens = ttsResp.OutputTokens
		usage.TotalTokens = ttsResp.TotalTokens
	}

	var audioBytes []byte
	if len(ttsResp.Choices) > 0 && ttsResp.Choices[0].Message.Audio.Data != "" {
		audioBytes, err = base64.StdEncoding.DecodeString(ttsResp.Choices[0].Message.Audio.Data)
		if err != nil {
			logger.LogError(c, fmt.Sprintf("failed to decode TTS audio: %v", err))
			c.Writer.WriteHeader(http.StatusInternalServerError)
			return usage
		}
	} else {
		logger.LogWarn(c, "mimo tts response contains no audio data")
	}

	c.Writer.Header().Set("Content-Type", getAudioContentType(audioFormat))
	c.Writer.Header().Set("Content-Length", fmt.Sprintf("%d", len(audioBytes)))
	c.Writer.WriteHeader(http.StatusOK)
	if len(audioBytes) > 0 {
		if _, err := c.Writer.Write(audioBytes); err != nil {
			logger.LogError(c, fmt.Sprintf("failed to write TTS response: %v", err))
		}
	}
	return usage
}

// ClaudeSTTHandler 处理 MiMo ASR 响应。
//
// MiMo ASR 复用 Chat Completions 协议，识别文本在 choices[0].message.content，
// 这里转换为 OpenAI 标准 Transcription 响应 {"text": "..."} 返回给客户端。
func ClaudeSTTHandler(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (*types.NewAPIError, *dto.Usage) {
	defer service.CloseResponseBodyGracefully(resp)

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return types.NewOpenAIError(err, types.ErrorCodeReadResponseBodyFailed, http.StatusInternalServerError), nil
	}

	var asrResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		dto.Usage `json:"usage"`
	}

	usage := &dto.Usage{}
	usage.PromptTokens = info.GetEstimatePromptTokens()

	responseText := ""
	if err := common.Unmarshal(bodyBytes, &asrResp); err == nil {
		if asrResp.TotalTokens != 0 {
			usage.PromptTokens = asrResp.InputTokens
			usage.CompletionTokens = asrResp.OutputTokens
		}
		if len(asrResp.Choices) > 0 {
			responseText = asrResp.Choices[0].Message.Content
		}
	} else {
		logger.LogError(c, fmt.Sprintf("failed to unmarshal ASR response: %v", err))
	}
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens

	// 构造 OpenAI 标准 Transcription 响应
	respBytes, _ := common.Marshal(dto.AudioResponse{Text: responseText})
	service.IOCopyBytesGracefully(c, resp, respBytes)

	return nil, usage
}

// getAudioContentType 根据音频格式返回对应的 MIME type。
func getAudioContentType(format string) string {
	switch format {
	case "wav":
		return "audio/wav"
	case "pcm":
		return "audio/pcm;rate=24000"
	case "opus":
		return "audio/opus"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	default:
		return "audio/mpeg"
	}
}
