<template>
  <div class="image-upload">
    <el-upload
      ref="uploadRef"
      class="upload-area"
      :class="{ 'has-image': previewUrl }"
      drag
      :auto-upload="false"
      :show-file-list="false"
      :accept="accept"
      :on-change="handleFileChange"
    >
      <div v-if="previewUrl" class="preview-container">
        <img :src="previewUrl" class="preview-image" />
        <div class="preview-overlay">
          <el-button type="primary" :icon="Upload" circle @click.stop />
          <el-button type="danger" :icon="Delete" circle @click.stop="clearImage" />
        </div>
      </div>
      <div v-else class="upload-placeholder">
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">
          <span class="primary-text">Drop image here or click to upload</span>
          <span class="secondary-text">Supports: JPG, PNG, WEBP (Max: {{ maxSizeMB }}MB)</span>
        </div>
      </div>
    </el-upload>
    
    <!-- Image Info -->
    <div v-if="imageInfo" class="image-info">
      <div class="info-item">
        <span class="label">Name:</span>
        <span class="value">{{ imageInfo.name }}</span>
      </div>
      <div class="info-item">
        <span class="label">Size:</span>
        <span class="value">{{ formatFileSize(imageInfo.size) }}</span>
      </div>
      <div class="info-item" v-if="imageInfo.dimensions">
        <span class="label">Dimensions:</span>
        <span class="value">{{ imageInfo.dimensions }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { ElMessage, type UploadFile, type UploadInstance } from 'element-plus'
import { UploadFilled, Upload, Delete } from '@element-plus/icons-vue'

interface ImageInfo {
  name: string
  size: number
  dimensions?: string
}

const props = withDefaults(defineProps<{
  accept?: string
  maxSizeMB?: number
  modelValue?: File | null
}>(), {
  accept: 'image/jpeg,image/png,image/webp',
  maxSizeMB: 10
})

const emit = defineEmits<{
  (e: 'update:modelValue', file: File | null): void
  (e: 'change', file: File | null): void
}>()

const uploadRef = ref<UploadInstance>()
const previewUrl = ref<string>('')
const imageInfo = ref<ImageInfo | null>(null)

const handleFileChange = (uploadFile: UploadFile) => {
  const file = uploadFile.raw
  if (!file) return
  
  // Validate file type
  if (!props.accept.includes(file.type)) {
    ElMessage.error('Invalid file type. Please upload an image.')
    return
  }
  
  // Validate file size
  const maxSize = props.maxSizeMB * 1024 * 1024
  if (file.size > maxSize) {
    ElMessage.error(`File size exceeds ${props.maxSizeMB}MB limit.`)
    return
  }
  
  // Create preview
  const reader = new FileReader()
  reader.onload = (e) => {
    previewUrl.value = e.target?.result as string
    
    // Get image dimensions
    const img = new Image()
    img.onload = () => {
      imageInfo.value = {
        name: file.name,
        size: file.size,
        dimensions: `${img.width} × ${img.height}`
      }
    }
    img.src = previewUrl.value
  }
  reader.readAsDataURL(file)
  
  emit('update:modelValue', file)
  emit('change', file)
}

const clearImage = () => {
  previewUrl.value = ''
  imageInfo.value = null
  uploadRef.value?.clearFiles()
  emit('update:modelValue', null)
  emit('change', null)
}

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
}

// Expose methods for parent component
defineExpose({
  clearImage
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.image-upload {
  width: 100%;
}

.upload-area {
  width: 100%;
  
  :deep(.el-upload) {
    width: 100%;
  }
  
  :deep(.el-upload-dragger) {
    width: 100%;
    height: 280px;
    @include glass-effect;
    border: 2px dashed rgba($neon-cyan, 0.3);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    
    &:hover {
      border-color: $neon-cyan;
      background: rgba($neon-cyan, 0.05);
    }
  }
  
  &.has-image {
    :deep(.el-upload-dragger) {
      padding: 0;
      overflow: hidden;
    }
  }
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.upload-icon {
  font-size: 64px;
  color: $neon-cyan;
  opacity: 0.8;
}

.upload-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  
  .primary-text {
    font-size: 16px;
    color: $text-primary;
    font-weight: 500;
  }
  
  .secondary-text {
    font-size: 13px;
    color: $text-muted;
  }
}

.preview-container {
  width: 100%;
  height: 100%;
  position: relative;
}

.preview-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: rgba(0, 0, 0, 0.3);
}

.preview-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  background: rgba(0, 0, 0, 0.5);
  opacity: 0;
  transition: opacity 0.3s ease;
  
  &:hover {
    opacity: 1;
  }
}

.image-info {
  margin-top: 16px;
  padding: 16px;
  @include glass-effect;
  border-radius: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.info-item {
  display: flex;
  align-items: center;
  gap: 8px;
  
  .label {
    font-size: 13px;
    color: $text-muted;
  }
  
  .value {
    font-size: 13px;
    color: $text-primary;
    font-weight: 500;
  }
}
</style>
