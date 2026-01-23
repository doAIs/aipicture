import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/text-to-image',
      name: 'text-to-image',
      component: () => import('../views/TextToImageView.vue')
    },
    {
      path: '/image-to-image',
      name: 'image-to-image',
      component: () => import('../views/ImageToImageView.vue')
    },
    {
      path: '/text-to-video',
      name: 'text-to-video',
      component: () => import('../views/TextToVideoView.vue')
    },
    {
      path: '/image-to-video',
      name: 'image-to-video',
      component: () => import('../views/ImageToVideoView.vue')
    },
    {
      path: '/video-to-video',
      name: 'video-to-video',
      component: () => import('../views/VideoToVideoView.vue')
    },
    {
      path: '/image-recognition',
      name: 'image-recognition',
      component: () => import('../views/ImageRecognitionView.vue')
    },
    {
      path: '/video-recognition',
      name: 'video-recognition',
      component: () => import('../views/VideoRecognitionView.vue')
    },
    {
      path: '/face-recognition',
      name: 'face-recognition',
      component: () => import('../views/FaceRecognitionView.vue')
    },
    {
      path: '/audio-processing',
      name: 'audio-processing',
      component: () => import('../views/AudioProcessingView.vue')
    },
    {
      path: '/model-training',
      name: 'model-training',
      component: () => import('../views/ModelTrainingView.vue')
    },
    {
      path: '/llm-finetuning',
      name: 'llm-finetuning',
      component: () => import('../views/LLMFineTuningView.vue')
    }
  ]
})

export default router