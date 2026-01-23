import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { trainingApi, llmFineTuningApi, type TrainingStatus, type TrainingConfig, type LoRAConfig } from '@/api/endpoints/training'
import { useWebSocket } from '@/api/websocket'

export interface TrainingTask extends TrainingStatus {
  type: 'model' | 'llm'
  config: TrainingConfig | LoRAConfig
}

export const useTrainingStore = defineStore('training', () => {
  // State
  const tasks = ref<TrainingTask[]>([])
  const currentTask = ref<TrainingTask | null>(null)
  const isTraining = ref(false)
  const error = ref<string | null>(null)

  // WebSocket for real-time updates
  const { connect, disconnect, on, state: wsState } = useWebSocket('training')

  // Getters
  const activeTasks = computed(() => 
    tasks.value.filter(t => t.status === 'running' || t.status === 'pending')
  )

  const completedTasks = computed(() => 
    tasks.value.filter(t => t.status === 'completed')
  )

  const failedTasks = computed(() => 
    tasks.value.filter(t => t.status === 'failed')
  )

  const overallProgress = computed(() => {
    if (activeTasks.value.length === 0) return 0
    const totalProgress = activeTasks.value.reduce((sum, task) => sum + (task.progress || 0), 0)
    return Math.round(totalProgress / activeTasks.value.length)
  })

  // Actions
  async function startModelTraining(config: TrainingConfig): Promise<string | null> {
    error.value = null
    isTraining.value = true

    try {
      const { task_id } = await trainingApi.startTraining(config)
      
      const newTask: TrainingTask = {
        task_id,
        type: 'model',
        config,
        status: 'pending',
        progress: 0
      }
      
      tasks.value.push(newTask)
      currentTask.value = newTask
      
      // Connect to WebSocket for real-time updates
      await connectWebSocket()
      
      return task_id
    } catch (e: any) {
      error.value = e.message || 'Failed to start training'
      isTraining.value = false
      return null
    }
  }

  async function startLLMFineTuning(config: LoRAConfig, useQLoRA = false): Promise<string | null> {
    error.value = null
    isTraining.value = true

    try {
      const { task_id } = useQLoRA 
        ? await llmFineTuningApi.startQLoRA(config)
        : await llmFineTuningApi.startLoRA(config)
      
      const newTask: TrainingTask = {
        task_id,
        type: 'llm',
        config,
        status: 'pending',
        progress: 0
      }
      
      tasks.value.push(newTask)
      currentTask.value = newTask
      
      // Connect to WebSocket for real-time updates
      await connectWebSocket()
      
      return task_id
    } catch (e: any) {
      error.value = e.message || 'Failed to start LLM fine-tuning'
      isTraining.value = false
      return null
    }
  }

  async function stopTraining(taskId: string) {
    try {
      const task = tasks.value.find(t => t.task_id === taskId)
      if (!task) return

      if (task.type === 'model') {
        await trainingApi.stopTraining(taskId)
      } else {
        await llmFineTuningApi.stopFineTuning(taskId)
      }

      task.status = 'cancelled'
      
      if (currentTask.value?.task_id === taskId) {
        currentTask.value = null
      }
      
      // Check if any tasks are still running
      if (activeTasks.value.length === 0) {
        isTraining.value = false
        disconnectWebSocket()
      }
    } catch (e: any) {
      error.value = e.message || 'Failed to stop training'
    }
  }

  async function fetchTaskStatus(taskId: string) {
    try {
      const task = tasks.value.find(t => t.task_id === taskId)
      if (!task) return

      let status: TrainingStatus
      if (task.type === 'model') {
        status = await trainingApi.getStatus(taskId)
      } else {
        status = await llmFineTuningApi.getStatus(taskId)
      }

      Object.assign(task, status)
      
      if (status.status === 'completed' || status.status === 'failed') {
        if (currentTask.value?.task_id === taskId) {
          currentTask.value = null
        }
        if (activeTasks.value.length === 0) {
          isTraining.value = false
        }
      }
    } catch (e: any) {
      error.value = e.message || 'Failed to fetch task status'
    }
  }

  async function fetchAllTasks() {
    try {
      const [modelTasks, llmTasks] = await Promise.all([
        trainingApi.listTasks(),
        llmFineTuningApi.listTasks()
      ])

      tasks.value = [
        ...modelTasks.map(t => ({ ...t, type: 'model' as const, config: {} as TrainingConfig })),
        ...llmTasks.map(t => ({ ...t, type: 'llm' as const, config: {} as LoRAConfig }))
      ]

      // Check if any tasks are running
      isTraining.value = activeTasks.value.length > 0
    } catch (e: any) {
      error.value = e.message || 'Failed to fetch tasks'
    }
  }

  async function connectWebSocket() {
    try {
      await connect()
      
      // Handle progress updates
      on('progress', (data: { task_id: string; progress: number; metrics?: Record<string, number> }) => {
        const task = tasks.value.find(t => t.task_id === data.task_id)
        if (task) {
          task.progress = data.progress
          if (data.metrics) {
            task.metrics = data.metrics
            task.loss = data.metrics.loss
          }
        }
      })

      // Handle status updates
      on('status', (data: TrainingStatus) => {
        const task = tasks.value.find(t => t.task_id === data.task_id)
        if (task) {
          Object.assign(task, data)
          
          if (data.status === 'completed' || data.status === 'failed') {
            if (currentTask.value?.task_id === data.task_id) {
              currentTask.value = null
            }
            if (activeTasks.value.length === 0) {
              isTraining.value = false
              disconnectWebSocket()
            }
          }
        }
      })

      // Handle errors
      on('error', (data: { task_id: string; message: string }) => {
        const task = tasks.value.find(t => t.task_id === data.task_id)
        if (task) {
          task.status = 'failed'
          task.error = data.message
        }
        error.value = data.message
      })
    } catch (e) {
      console.error('Failed to connect WebSocket:', e)
    }
  }

  function disconnectWebSocket() {
    disconnect()
  }

  function clearError() {
    error.value = null
  }

  function removeTask(taskId: string) {
    const index = tasks.value.findIndex(t => t.task_id === taskId)
    if (index > -1) {
      tasks.value.splice(index, 1)
    }
  }

  return {
    // State
    tasks,
    currentTask,
    isTraining,
    error,
    wsState,
    // Getters
    activeTasks,
    completedTasks,
    failedTasks,
    overallProgress,
    // Actions
    startModelTraining,
    startLLMFineTuning,
    stopTraining,
    fetchTaskStatus,
    fetchAllTasks,
    connectWebSocket,
    disconnectWebSocket,
    clearError,
    removeTask
  }
})
