<template>
  <aside class="app-sidebar" :class="{ collapsed: isCollapsed }">
    <el-menu
      :default-active="activeMenu"
      :collapse="isCollapsed"
      :collapse-transition="false"
      router
      class="sidebar-menu"
    >
      <!-- Dashboard -->
      <el-menu-item index="/">
        <el-icon><HomeFilled /></el-icon>
        <template #title>Dashboard</template>
      </el-menu-item>

      <!-- Generation Section -->
      <el-sub-menu index="generation">
        <template #title>
          <el-icon><MagicStick /></el-icon>
          <span>Generation</span>
        </template>
        <el-menu-item index="/text-to-image">
          <el-icon><Picture /></el-icon>
          <template #title>Text to Image</template>
        </el-menu-item>
        <el-menu-item index="/image-to-image">
          <el-icon><PictureFilled /></el-icon>
          <template #title>Image to Image</template>
        </el-menu-item>
        <el-menu-item index="/text-to-video">
          <el-icon><VideoCamera /></el-icon>
          <template #title>Text to Video</template>
        </el-menu-item>
        <el-menu-item index="/image-to-video">
          <el-icon><Film /></el-icon>
          <template #title>Image to Video</template>
        </el-menu-item>
        <el-menu-item index="/video-to-video">
          <el-icon><VideoCameraFilled /></el-icon>
          <template #title>Video to Video</template>
        </el-menu-item>
      </el-sub-menu>

      <!-- Recognition Section -->
      <el-sub-menu index="recognition">
        <template #title>
          <el-icon><View /></el-icon>
          <span>Recognition</span>
        </template>
        <el-menu-item index="/image-recognition">
          <el-icon><Search /></el-icon>
          <template #title>Image Recognition</template>
        </el-menu-item>
        <el-menu-item index="/video-recognition">
          <el-icon><Monitor /></el-icon>
          <template #title>Video Recognition</template>
        </el-menu-item>
        <el-menu-item index="/face-recognition">
          <el-icon><Avatar /></el-icon>
          <template #title>Face Recognition</template>
        </el-menu-item>
      </el-sub-menu>

      <!-- Audio Section -->
      <el-menu-item index="/audio-processing">
        <el-icon><Microphone /></el-icon>
        <template #title>Audio Processing</template>
      </el-menu-item>

      <!-- Training Section -->
      <el-sub-menu index="training">
        <template #title>
          <el-icon><DataAnalysis /></el-icon>
          <span>Training</span>
        </template>
        <el-menu-item index="/model-training">
          <el-icon><SetUp /></el-icon>
          <template #title>Model Training</template>
        </el-menu-item>
        <el-menu-item index="/llm-finetuning">
          <el-icon><Cpu /></el-icon>
          <template #title>LLM Fine-tuning</template>
        </el-menu-item>
      </el-sub-menu>
    </el-menu>

    <!-- Sidebar Footer -->
    <div class="sidebar-footer" v-if="!isCollapsed">
      <div class="footer-info">
        <div class="version">v2.0.0</div>
        <div class="copyright">AI Multimedia Platform</div>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import {
  HomeFilled,
  MagicStick,
  Picture,
  PictureFilled,
  VideoCamera,
  VideoCameraFilled,
  Film,
  View,
  Search,
  Monitor,
  Avatar,
  Microphone,
  DataAnalysis,
  SetUp,
  Cpu
} from '@element-plus/icons-vue'

defineProps<{
  isCollapsed: boolean
}>()

const route = useRoute()

const activeMenu = computed(() => {
  return route.path
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.app-sidebar {
  width: 260px;
  height: calc(100vh - 64px);
  position: fixed;
  top: 64px;
  left: 0;
  background: rgba($bg-secondary, 0.95);
  backdrop-filter: blur(10px);
  border-right: 1px solid $glass-border;
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  z-index: 999;
  overflow: hidden;
  
  &.collapsed {
    width: 80px;
    
    .sidebar-footer {
      display: none;
    }
  }
}

.sidebar-menu {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 12px 0;
  background: transparent !important;
  border: none !important;
  
  @include tech-scrollbar;
  
  :deep(.el-menu-item),
  :deep(.el-sub-menu__title) {
    height: 48px;
    line-height: 48px;
    margin: 4px 12px;
    border-radius: 8px;
    color: $text-secondary !important;
    transition: all 0.3s ease;
    
    .el-icon {
      font-size: 18px;
      color: inherit;
      margin-right: 12px;
    }
    
    &:hover {
      background: rgba($neon-cyan, 0.1) !important;
      color: $neon-cyan !important;
    }
  }
  
  :deep(.el-menu-item.is-active) {
    background: linear-gradient(90deg, rgba($neon-cyan, 0.2), transparent) !important;
    color: $neon-cyan !important;
    border-left: 3px solid $neon-cyan;
    margin-left: 9px;
    
    &::before {
      content: '';
      position: absolute;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 3px;
      height: 60%;
      background: $neon-cyan;
      box-shadow: 0 0 10px $neon-cyan;
      border-radius: 0 2px 2px 0;
    }
  }
  
  :deep(.el-sub-menu) {
    .el-sub-menu__title {
      &:hover {
        background: rgba($neon-cyan, 0.1) !important;
      }
    }
    
    .el-menu {
      background: transparent !important;
      
      .el-menu-item {
        padding-left: 56px !important;
        margin: 2px 12px 2px 24px;
        height: 42px;
        line-height: 42px;
      }
    }
  }
  
  :deep(.el-sub-menu__icon-arrow) {
    color: $text-muted;
  }
  
  // Collapsed state styles
  &.el-menu--collapse {
    :deep(.el-menu-item),
    :deep(.el-sub-menu__title) {
      margin: 4px 8px;
      justify-content: center;
      
      .el-icon {
        margin: 0;
      }
    }
    
    :deep(.el-sub-menu) {
      .el-sub-menu__title {
        padding: 0 !important;
        justify-content: center;
      }
    }
  }
}

.sidebar-footer {
  padding: 16px 20px;
  border-top: 1px solid $glass-border;
  background: rgba(0, 0, 0, 0.2);
}

.footer-info {
  text-align: center;
  
  .version {
    font-size: 12px;
    color: $neon-cyan;
    margin-bottom: 4px;
    font-weight: 600;
  }
  
  .copyright {
    font-size: 11px;
    color: $text-muted;
  }
}
</style>
