import api from '../index'

// Types
export interface TrainingConfig {
  model_type: 'image_classifier' | 'object_detector' | 'text_to_image' | 'llm'
  base_model: string
  dataset_path?: string
  output_dir?: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  save_steps?: number
  eval_steps?: number
  warmup_steps?: number
  max_grad_norm?: number
  weight_decay?: number
  seed?: number
}

export interface LoRAConfig {
  base_model: string
  dataset_path?: string
  output_dir?: string
  lora_r?: number
  lora_alpha?: number
  lora_dropout?: number
  target_modules?: string[]
  epochs?: number
  batch_size?: number
  learning_rate?: number
  max_length?: number
  gradient_accumulation_steps?: number
  use_4bit?: boolean  // QLoRA
  use_8bit?: boolean
}

export interface TrainingStatus {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_epoch?: number
  total_epochs?: number
  current_step?: number
  total_steps?: number
  loss?: number
  learning_rate?: number
  metrics?: Record<string, number>
  error?: string
  started_at?: string
  completed_at?: string
}

export interface DatasetInfo {
  id: string
  name: string
  type: string
  size: number
  samples: number
  created_at: string
}

// Model Training API
export const trainingApi = {
  // Start training
  startTraining: (config: TrainingConfig): Promise<{ task_id: string }> => {
    return api.post('/training/start', config)
  },
  
  // Get training status
  getStatus: (taskId: string): Promise<TrainingStatus> => {
    return api.get(`/training/status/${taskId}`)
  },
  
  // Stop training
  stopTraining: (taskId: string): Promise<void> => {
    return api.post(`/training/stop/${taskId}`)
  },
  
  // List all training tasks
  listTasks: (): Promise<TrainingStatus[]> => {
    return api.get('/training/tasks')
  },
  
  // Upload dataset
  uploadDataset: (file: File, name: string, type: string, onProgress?: (percent: number) => void): Promise<DatasetInfo> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('name', name)
    formData.append('type', type)
    
    return api.upload('/training/datasets/upload', formData, onProgress)
  },
  
  // List datasets
  listDatasets: (): Promise<DatasetInfo[]> => {
    return api.get('/training/datasets')
  },
  
  // Delete dataset
  deleteDataset: (datasetId: string): Promise<void> => {
    return api.delete(`/training/datasets/${datasetId}`)
  },
  
  // Get available base models
  getBaseModels: (modelType: string): Promise<{ id: string; name: string; description: string }[]> => {
    return api.get('/training/models', { model_type: modelType })
  },
  
  // Get training history
  getHistory: (): Promise<TrainingStatus[]> => {
    return api.get('/training/history')
  },
  
  // Download trained model
  downloadModel: (taskId: string): Promise<void> => {
    return api.download(`/training/download/${taskId}`, `model_${taskId}.zip`)
  }
}

// LLM Fine-tuning API
export const llmFineTuningApi = {
  // Start LoRA training
  startLoRA: (config: LoRAConfig): Promise<{ task_id: string }> => {
    return api.post('/llm-finetuning/lora/start', config)
  },
  
  // Start QLoRA training (4-bit quantized)
  startQLoRA: (config: LoRAConfig): Promise<{ task_id: string }> => {
    return api.post('/llm-finetuning/qlora/start', { ...config, use_4bit: true })
  },
  
  // Get fine-tuning status
  getStatus: (taskId: string): Promise<TrainingStatus> => {
    return api.get(`/llm-finetuning/status/${taskId}`)
  },
  
  // Stop fine-tuning
  stopFineTuning: (taskId: string): Promise<void> => {
    return api.post(`/llm-finetuning/stop/${taskId}`)
  },
  
  // List all fine-tuning tasks
  listTasks: (): Promise<TrainingStatus[]> => {
    return api.get('/llm-finetuning/tasks')
  },
  
  // Get available base LLM models
  getBaseModels: (): Promise<{ id: string; name: string; parameters: string; description: string }[]> => {
    return api.get('/llm-finetuning/models')
  },
  
  // Upload training data (JSONL format)
  uploadTrainingData: (file: File, onProgress?: (percent: number) => void): Promise<{ path: string; samples: number }> => {
    const formData = new FormData()
    formData.append('file', file)
    
    return api.upload('/llm-finetuning/data/upload', formData, onProgress)
  },
  
  // Merge LoRA adapter with base model
  mergeAdapter: (taskId: string, outputPath: string): Promise<{ success: boolean; path: string }> => {
    return api.post('/llm-finetuning/merge', { task_id: taskId, output_path: outputPath })
  },
  
  // Test fine-tuned model
  testModel: (taskId: string, prompt: string, maxTokens?: number): Promise<{ response: string }> => {
    return api.post('/llm-finetuning/test', { task_id: taskId, prompt, max_tokens: maxTokens })
  },
  
  // Export adapter
  exportAdapter: (taskId: string): Promise<void> => {
    return api.download(`/llm-finetuning/export/${taskId}`, `lora_adapter_${taskId}.zip`)
  }
}

export default {
  training: trainingApi,
  llmFineTuning: llmFineTuningApi
}
