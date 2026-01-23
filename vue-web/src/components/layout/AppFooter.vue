<template>
  <footer class="app-footer">
    <div class="footer-content">
      <div class="footer-left">
        <span class="copyright">© 2024 AI Multimedia Platform</span>
        <span class="divider">|</span>
        <span class="version">Version 2.0.0</span>
      </div>
      
      <div class="footer-center">
        <div class="status-indicators">
          <div class="indicator" v-for="(status, key) in systemStatus" :key="key">
            <span class="indicator-dot" :class="status.status"></span>
            <span class="indicator-label">{{ status.label }}</span>
          </div>
        </div>
      </div>
      
      <div class="footer-right">
        <el-tooltip content="GitHub Repository" placement="top">
          <el-button :icon="Link" circle size="small" class="footer-btn" />
        </el-tooltip>
        <el-tooltip content="Documentation" placement="top">
          <el-button :icon="Document" circle size="small" class="footer-btn" />
        </el-tooltip>
        <el-tooltip content="API Reference" placement="top">
          <el-button :icon="Connection" circle size="small" class="footer-btn" />
        </el-tooltip>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { Link, Document, Connection } from '@element-plus/icons-vue'

const systemStatus = reactive({
  backend: {
    label: 'Backend',
    status: 'online'
  },
  gpu: {
    label: 'GPU',
    status: 'online'
  },
  models: {
    label: 'Models',
    status: 'online'
  }
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.app-footer {
  height: 48px;
  background: rgba($bg-secondary, 0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid $glass-border;
  position: fixed;
  bottom: 0;
  left: 260px;
  right: 0;
  z-index: 998;
  transition: left 0.3s ease;
}

.footer-content {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
}

.footer-left {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: $text-muted;
  
  .copyright {
    color: $text-secondary;
  }
  
  .divider {
    color: $glass-border;
  }
  
  .version {
    color: $neon-cyan;
  }
}

.footer-center {
  .status-indicators {
    display: flex;
    gap: 24px;
  }
  
  .indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: $text-secondary;
  }
  
  .indicator-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    
    &.online {
      background: $neon-green;
      box-shadow: 0 0 6px $neon-green;
    }
    
    &.offline {
      background: #ff4757;
      box-shadow: 0 0 6px #ff4757;
    }
    
    &.loading {
      background: $neon-yellow;
      box-shadow: 0 0 6px $neon-yellow;
      animation: pulse-glow 1.5s infinite;
    }
  }
}

.footer-right {
  display: flex;
  gap: 8px;
  
  .footer-btn {
    background: transparent !important;
    border: 1px solid $glass-border !important;
    color: $text-secondary !important;
    
    &:hover {
      border-color: $neon-cyan !important;
      color: $neon-cyan !important;
    }
  }
}

// Responsive
@media (max-width: 992px) {
  .app-footer {
    left: 0;
  }
  
  .footer-center {
    display: none;
  }
}

@media (max-width: 768px) {
  .footer-left {
    .divider,
    .version {
      display: none;
    }
  }
}
</style>
