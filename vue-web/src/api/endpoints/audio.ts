import api from '../index'

// Types
export interface TranscriptionResult {
  task_id: string
  status: string
  text?: string
  segments?: TranscriptionSegment[]
  language?: string
  duration?: number
  error?: string
}

export interface TranscriptionSegment {
  id: number
  start: number
  end: number
  text: string
  confidence?: number
}

export interface TTSParams {
  text: string
  voice?: string
  language?: string
  speed?: number
  pitch?: number
}

export interface TTSResult {
  task_id: string
  status: string
  audio_url?: string
  duration?: number
  error?: string
}

export interface VoiceInfo {
  id: string
  name: string
  language: string
  gender: string
  preview_url?: string
}

// Audio Processing API
export const audioApi = {
  // Speech-to-Text (Transcription)
  transcribe: (audio: File, language?: string, model?: string): Promise<TranscriptionResult> => {
    const formData = new FormData()
    formData.append('audio', audio)
    if (language) formData.append('language', language)
    if (model) formData.append('model', model)
    
    return api.upload('/audio/transcribe', formData)
  },
  
  // Get transcription status
  getTranscriptionStatus: (taskId: string): Promise<TranscriptionResult> => {
    return api.get(`/audio/transcribe/status/${taskId}`)
  },
  
  // Text-to-Speech
  synthesize: (params: TTSParams): Promise<TTSResult> => {
    return api.post('/audio/synthesize', params)
  },
  
  // Get TTS status
  getTTSStatus: (taskId: string): Promise<TTSResult> => {
    return api.get(`/audio/synthesize/status/${taskId}`)
  },
  
  // Get available voices
  getVoices: (language?: string): Promise<VoiceInfo[]> => {
    return api.get('/audio/voices', language ? { language } : undefined)
  },
  
  // Get available transcription models
  getTranscriptionModels: (): Promise<{ id: string; name: string; languages: string[] }[]> => {
    return api.get('/audio/models/transcription')
  },
  
  // Get available TTS models
  getTTSModels: (): Promise<{ id: string; name: string; description: string }[]> => {
    return api.get('/audio/models/tts')
  },
  
  // Audio classification
  classify: (audio: File): Promise<{ classifications: { label: string; confidence: number }[] }> => {
    const formData = new FormData()
    formData.append('audio', audio)
    
    return api.upload('/audio/classify', formData)
  },
  
  // Voice cloning (if supported)
  cloneVoice: (audio: File, name: string): Promise<{ voice_id: string; name: string }> => {
    const formData = new FormData()
    formData.append('audio', audio)
    formData.append('name', name)
    
    return api.upload('/audio/voices/clone', formData)
  },
  
  // Delete cloned voice
  deleteVoice: (voiceId: string): Promise<void> => {
    return api.delete(`/audio/voices/${voiceId}`)
  }
}

export default audioApi
