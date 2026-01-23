<template>
  <header class="app-header">
    <div class="header-left">
      <el-button 
        :icon="isCollapsed ? Expand : Fold" 
        circle 
        class="toggle-btn"
        @click="toggleSidebar"
      />
      <div class="logo">
        <span class="logo-icon">🤖</span>
        <span class="logo-text" v-if="!isCollapsed">AI Platform</span>
      </div>
    </div>
    
    <div class="header-center">
      <el-breadcrumb separator="/">
        <el-breadcrumb-item :to="{ path: '/' }">Home</el-breadcrumb-item>
        <el-breadcrumb-item v-if="currentRoute">{{ currentRoute }}</el-breadcrumb-item>
      </el-breadcrumb>
    </div>
    
    <div class="header-right">
      <div class="system-status">
        <div class="status-item">
          <span class="status-dot online"></span>
          <span class="status-label">System Online</span>
        </div>
        <div class="status-item" v-if="gpuStatus">
          <el-icon><Monitor /></el-icon>
          <span class="status-label">GPU: {{ gpuStatus }}</span>
        </div>
      </div>
      
      <el-dropdown trigger="click" class="user-menu">
        <div class="user-avatar">
          <el-avatar :size="36" src="" class="avatar">
            <el-icon><User /></el-icon>
          </el-avatar>
        </div>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item :icon="Setting">Settings</el-dropdown-item>
            <el-dropdown-item :icon="Document">Documentation</el-dropdown-item>
            <el-dropdown-item divided :icon="SwitchButton">Logout</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { 
  Fold, 
  Expand, 
  Monitor, 
  User, 
  Setting, 
  Document, 
  SwitchButton 
} from '@element-plus/icons-vue'

const props = defineProps<{
  isCollapsed: boolean
  gpuStatus?: string
}>()

const emit = defineEmits<{
  (e: 'toggle-sidebar'): void
}>()

const route = useRoute()

const currentRoute = computed(() => {
  const name = route.name as string
  if (!name || name === 'home') return ''
  return name.split('-').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ')
})

const toggleSidebar = () => {
  emit('toggle-sidebar')
}
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.app-header {
  height: 64px;
  background: rgba($bg-secondary, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid $glass-border;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.toggle-btn {
  background: transparent !important;
  border: 1px solid $glass-border !important;
  color: $text-secondary !important;
  
  &:hover {
    border-color: $neon-cyan !important;
    color: $neon-cyan !important;
  }
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  
  .logo-icon {
    font-size: 24px;
  }
  
  .logo-text {
    font-size: 18px;
    font-weight: 700;
    background: $gradient-neon;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
}

.header-center {
  flex: 1;
  display: flex;
  justify-content: center;
  
  :deep(.el-breadcrumb) {
    .el-breadcrumb__item {
      .el-breadcrumb__inner {
        color: $text-secondary;
        
        &:hover {
          color: $neon-cyan;
        }
      }
      
      &:last-child .el-breadcrumb__inner {
        color: $neon-cyan;
      }
    }
    
    .el-breadcrumb__separator {
      color: $text-muted;
    }
  }
}

.header-right {
  display: flex;
  align-items: center;
  gap: 24px;
}

.system-status {
  display: flex;
  align-items: center;
  gap: 20px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: $text-secondary;
  
  .el-icon {
    color: $neon-cyan;
  }
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  
  &.online {
    background: $neon-green;
    box-shadow: 0 0 8px $neon-green;
  }
  
  &.offline {
    background: $neon-orange;
    box-shadow: 0 0 8px $neon-orange;
  }
}

.user-menu {
  cursor: pointer;
}

.user-avatar {
  .avatar {
    background: linear-gradient(135deg, $neon-cyan, $neon-purple);
    border: 2px solid transparent;
    transition: all 0.3s ease;
    
    &:hover {
      border-color: $neon-cyan;
      box-shadow: 0 0 15px rgba($neon-cyan, 0.5);
    }
  }
}
</style>
