import api from '../index'

// Types
export interface GenerationParams {
  prompt: string
  negative_prompt?: string
  width?: number
  height?: number
  num_inference_steps?: number
  guidance_scale?: number
  seed?: number
  model_id?: string
}

export interface GenerationResponse {
  task_id: string
  status: string
  image_url?: string
  video_url?: string
  progress?: number
  error?: string
}

export interface ImageToImageParams extends GenerationParams {
  image: File
  strength?: number
}

export interface VideoParams extends GenerationParams {
  num_frames?: number
  fps?: number
}

// Text-to-Image API
export const textToImageApi = {
  generate: (params: GenerationParams): Promise<GenerationResponse> => {
    return api.post('/text-to-image/generate', params)
  },
  
  getStatus: (taskId: string): Promise<GenerationResponse> => {
    return api.get(`/text-to-image/status/${taskId}`)
  },
  
  getHistory: (): Promise<GenerationResponse[]> => {
    return api.get('/text-to-image/history')
  },
  
  getModels: (): Promise<{ id: string; name: string; description: string }[]> => {
    return api.get('/text-to-image/models')
  }
}

// Image-to-Image API
export const imageToImageApi = {
  generate: (params: ImageToImageParams): Promise<GenerationResponse> => {
    const formData = new FormData()
    formData.append('image', params.image)
    formData.append('prompt', params.prompt)
    if (params.negative_prompt) formData.append('negative_prompt', params.negative_prompt)
    if (params.strength) formData.append('strength', params.strength.toString())
    if (params.num_inference_steps) formData.append('num_inference_steps', params.num_inference_steps.toString())
    if (params.guidance_scale) formData.append('guidance_scale', params.guidance_scale.toString())
    if (params.seed) formData.append('seed', params.seed.toString())
    if (params.model_id) formData.append('model_id', params.model_id)
    
    return api.upload('/image-to-image/generate', formData)
  },
  
  getStatus: (taskId: string): Promise<GenerationResponse> => {
    return api.get(`/image-to-image/status/${taskId}`)
  }
}

// Text-to-Video API
export const textToVideoApi = {
  generate: (params: VideoParams): Promise<GenerationResponse> => {
    return api.post('/text-to-video/generate', params)
  },
  
  getStatus: (taskId: string): Promise<GenerationResponse> => {
    return api.get(`/text-to-video/status/${taskId}`)
  },
  
  getModels: (): Promise<{ id: string; name: string; description: string }[]> => {
    return api.get('/text-to-video/models')
  }
}

// Image-to-Video API
export const imageToVideoApi = {
  generate: (image: File, params: VideoParams): Promise<GenerationResponse> => {
    const formData = new FormData()
    formData.append('image', image)
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) formData.append(key, value.toString())
    })
    
    return api.upload('/image-to-video/generate', formData)
  },
  
  getStatus: (taskId: string): Promise<GenerationResponse> => {
    return api.get(`/image-to-video/status/${taskId}`)
  }
}

// Video-to-Video API
export const videoToVideoApi = {
  generate: (video: File, params: GenerationParams): Promise<GenerationResponse> => {
    const formData = new FormData()
    formData.append('video', video)
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) formData.append(key, value.toString())
    })
    
    return api.upload('/video-to-video/generate', formData)
  },
  
  getStatus: (taskId: string): Promise<GenerationResponse> => {
    return api.get(`/video-to-video/status/${taskId}`)
  }
}

export default {
  textToImage: textToImageApi,
  imageToImage: imageToImageApi,
  textToVideo: textToVideoApi,
  imageToVideo: imageToVideoApi,
  videoToVideo: videoToVideoApi
}
