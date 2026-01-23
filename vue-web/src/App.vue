<template>
  <div class="app-wrapper">
    <AppHeader 
      :is-collapsed="sidebarCollapsed" 
      :gpu-status="gpuStatus"
      @toggle-sidebar="toggleSidebar"
    />
    <AppSidebar :is-collapsed="sidebarCollapsed" />
    <main class="main-content" :class="{ collapsed: sidebarCollapsed }">
      <div class="page-wrapper">
        <RouterView />
      </div>
    </main>
    <AppFooter />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterView } from 'vue-router'
import { AppHeader, AppSidebar, AppFooter } from '@/components/layout'

const sidebarCollapsed = ref(false)
const gpuStatus = ref('Available')

const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

onMounted(() => {
  // Check system status on mount
  checkSystemStatus()
})

const checkSystemStatus = async () => {
  // TODO: Implement API call to check GPU status
  gpuStatus.value = 'CUDA Available'
}
</script>

<style lang="scss">
@import '@/styles/main.scss';

.app-wrapper {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-content {
  margin-left: 260px;
  margin-top: 64px;
  margin-bottom: 48px;
  min-height: calc(100vh - 112px);
  padding: 24px;
  transition: margin-left 0.3s ease;
  
  &.collapsed {
    margin-left: 80px;
  }
}

.page-wrapper {
  max-width: 1600px;
  margin: 0 auto;
}

// Responsive adjustments
@media (max-width: 992px) {
  .main-content {
    margin-left: 0;
    padding: 16px;
  }
}
</style>
