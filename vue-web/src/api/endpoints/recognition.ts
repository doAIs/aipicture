import api from '../index'

// Types
export interface RecognitionResult {
  task_id: string
  status: string
  detections?: Detection[]
  classifications?: Classification[]
  faces?: FaceDetection[]
  error?: string
}

export interface Detection {
  label: string
  confidence: number
  bbox: [number, number, number, number]  // x, y, width, height
}

export interface Classification {
  label: string
  confidence: number
}

export interface FaceDetection {
  face_id: string
  name?: string
  confidence: number
  bbox: [number, number, number, number]
  landmarks?: Record<string, [number, number]>
  encoding?: number[]
}

export interface FaceRegisterParams {
  name: string
  image: File
}

// Image Recognition API
export const imageRecognitionApi = {
  classify: (image: File, model?: string): Promise<RecognitionResult> => {
    const formData = new FormData()
    formData.append('image', image)
    if (model) formData.append('model', model)
    
    return api.upload('/image-recognition/classify', formData)
  },
  
  detect: (image: File, model?: string): Promise<RecognitionResult> => {
    const formData = new FormData()
    formData.append('image', image)
    if (model) formData.append('model', model)
    
    return api.upload('/image-recognition/detect', formData)
  },
  
  getModels: (): Promise<{ id: string; name: string; type: string }[]> => {
    return api.get('/image-recognition/models')
  }
}

// Video Recognition API
export const videoRecognitionApi = {
  analyze: (video: File, model?: string): Promise<RecognitionResult> => {
    const formData = new FormData()
    formData.append('video', video)
    if (model) formData.append('model', model)
    
    return api.upload('/video-recognition/analyze', formData)
  },
  
  getStatus: (taskId: string): Promise<RecognitionResult> => {
    return api.get(`/video-recognition/status/${taskId}`)
  },
  
  startLiveDetection: (source: string): Promise<{ session_id: string }> => {
    return api.post('/video-recognition/live/start', { source })
  },
  
  stopLiveDetection: (sessionId: string): Promise<void> => {
    return api.post('/video-recognition/live/stop', { session_id: sessionId })
  }
}

// Face Recognition API
export const faceRecognitionApi = {
  detect: (image: File): Promise<RecognitionResult> => {
    const formData = new FormData()
    formData.append('image', image)
    
    return api.upload('/face-recognition/detect', formData)
  },
  
  recognize: (image: File): Promise<RecognitionResult> => {
    const formData = new FormData()
    formData.append('image', image)
    
    return api.upload('/face-recognition/recognize', formData)
  },
  
  register: (params: FaceRegisterParams): Promise<{ success: boolean; face_id: string }> => {
    const formData = new FormData()
    formData.append('name', params.name)
    formData.append('image', params.image)
    
    return api.upload('/face-recognition/register', formData)
  },
  
  deleteFace: (faceId: string): Promise<void> => {
    return api.delete(`/face-recognition/faces/${faceId}`)
  },
  
  listFaces: (): Promise<{ id: string; name: string; created_at: string }[]> => {
    return api.get('/face-recognition/faces')
  },
  
  startLiveRecognition: (): Promise<{ session_id: string }> => {
    return api.post('/face-recognition/live/start')
  },
  
  stopLiveRecognition: (sessionId: string): Promise<void> => {
    return api.post('/face-recognition/live/stop', { session_id: sessionId })
  }
}

export default {
  imageRecognition: imageRecognitionApi,
  videoRecognition: videoRecognitionApi,
  faceRecognition: faceRecognitionApi
}
