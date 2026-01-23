import { ref, type Ref } from 'vue'
import { ElNotification } from 'element-plus'

// WebSocket Configuration
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'

// WebSocket connection states
export type WebSocketState = 'connecting' | 'connected' | 'disconnected' | 'error'

// Message types
export interface WebSocketMessage {
  type: string
  data: any
  timestamp?: number
}

// WebSocket manager class
export class WebSocketManager {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts: number = 0
  private maxReconnectAttempts: number = 5
  private reconnectDelay: number = 3000
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map()
  
  public state: Ref<WebSocketState> = ref('disconnected')
  public lastMessage: Ref<WebSocketMessage | null> = ref(null)
  public error: Ref<string | null> = ref(null)
  
  constructor(endpoint: string) {
    this.url = `${WS_BASE_URL}/ws/${endpoint}`
  }
  
  // Connect to WebSocket
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }
      
      this.state.value = 'connecting'
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        this.state.value = 'connected'
        this.reconnectAttempts = 0
        this.error.value = null
        this.startHeartbeat()
        resolve()
      }
      
      this.ws.onclose = (event) => {
        this.state.value = 'disconnected'
        this.stopHeartbeat()
        
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++
          setTimeout(() => this.connect(), this.reconnectDelay)
        }
      }
      
      this.ws.onerror = (error) => {
        this.state.value = 'error'
        this.error.value = 'WebSocket connection error'
        reject(error)
      }
      
      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          message.timestamp = Date.now()
          this.lastMessage.value = message
          
          // Dispatch to handlers
          const handlers = this.messageHandlers.get(message.type)
          if (handlers) {
            handlers.forEach(handler => handler(message.data))
          }
          
          // Also dispatch to 'all' handlers
          const allHandlers = this.messageHandlers.get('*')
          if (allHandlers) {
            allHandlers.forEach(handler => handler(message))
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
    })
  }
  
  // Disconnect from WebSocket
  disconnect(): void {
    this.stopHeartbeat()
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.state.value = 'disconnected'
  }
  
  // Send message through WebSocket
  send(type: string, data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }))
    } else {
      console.error('WebSocket is not connected')
    }
  }
  
  // Subscribe to message type
  on(type: string, handler: (data: any) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, [])
    }
    this.messageHandlers.get(type)!.push(handler)
    
    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type)
      if (handlers) {
        const index = handlers.indexOf(handler)
        if (index > -1) {
          handlers.splice(index, 1)
        }
      }
    }
  }
  
  // Remove all handlers for a type
  off(type: string): void {
    this.messageHandlers.delete(type)
  }
  
  // Clear all handlers
  offAll(): void {
    this.messageHandlers.clear()
  }
  
  // Start heartbeat
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30000) // Every 30 seconds
  }
  
  // Stop heartbeat
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }
}

// Pre-configured WebSocket instances for different purposes
export const trainingWS = new WebSocketManager('training')
export const cameraWS = new WebSocketManager('camera')
export const generationWS = new WebSocketManager('generation')

// Composable for using WebSocket in components
export function useWebSocket(endpoint: string) {
  const manager = new WebSocketManager(endpoint)
  
  const connect = async () => {
    try {
      await manager.connect()
    } catch (error) {
      ElNotification({
        title: 'Connection Error',
        message: 'Failed to establish real-time connection',
        type: 'error'
      })
    }
  }
  
  const disconnect = () => {
    manager.disconnect()
  }
  
  const send = (type: string, data: any) => {
    manager.send(type, data)
  }
  
  const on = (type: string, handler: (data: any) => void) => {
    return manager.on(type, handler)
  }
  
  return {
    state: manager.state,
    lastMessage: manager.lastMessage,
    error: manager.error,
    connect,
    disconnect,
    send,
    on
  }
}

export default WebSocketManager
