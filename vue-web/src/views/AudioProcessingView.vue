<template>
  <div class="audio-processing-view">
    <div class="page-header">
      <h1>Audio Processing</h1>
      <p>Speech-to-text transcription and text-to-speech synthesis</p>
    </div>

    <el-tabs v-model="activeTab" class="audio-tabs">
      <!-- Speech to Text Tab -->
      <el-tab-pane label="Speech to Text" name="stt">
        <div class="content-grid">
          <div class="input-panel">
            <div class="panel-card">
              <h3>Audio Input</h3>
              <el-radio-group v-model="sttInputType" class="input-selector">
                <el-radio-button label="upload">Upload File</el-radio-button>
                <el-radio-button label="record">Record</el-radio-button>
              </el-radio-group>
              
              <div v-if="sttInputType === 'upload'" class="upload-area">
                <el-upload
                  drag
                  :auto-upload="false"
                  :on-change="handleAudioUpload"
                  accept="audio/*"
                  class="audio-upload"
                >
                  <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                  <div class="el-upload__text">
                    Drop audio file or click to upload<br>
                    <em>MP3, WAV, M4A supported</em>
                  </div>
                </el-upload>
                
                <div v-if="audioFile" class="audio-preview">
                  <audio :src="audioUrl" controls />
                </div>
              </div>
              
              <div v-else class="record-area">
                <div class="recording-visualizer" :class="{ active: isRecording }">
                  <div class="visualizer-bar" v-for="i in 20" :key="i" />
                </div>
                <el-button 
                  :type="isRecording ? 'danger' : 'primary'"
                  size="large"
                  circle
                  @click="toggleRecording"
                >
                  <el-icon><Microphone /></el-icon>
                </el-button>
                <span class="recording-time">{{ recordingTime }}</span>
              </div>
            </div>

            <div class="panel-card">
              <h3>Settings</h3>
              <div class="setting-row">
                <label>Model</label>
                <el-select v-model="sttModel" class="full-width">
                  <el-option label="Whisper Large" value="whisper-large" />
                  <el-option label="Whisper Medium" value="whisper-medium" />
                  <el-option label="Whisper Small" value="whisper-small" />
                </el-select>
              </div>
              <div class="setting-row">
                <label>Language</label>
                <el-select v-model="sttLanguage" class="full-width">
                  <el-option label="Auto Detect" value="auto" />
                  <el-option label="English" value="en" />
                  <el-option label="Chinese" value="zh" />
                  <el-option label="Japanese" value="ja" />
                  <el-option label="Korean" value="ko" />
                </el-select>
              </div>
            </div>

            <el-button 
              type="primary" 
              size="large"
              :loading="isTranscribing"
              :disabled="!audioFile && !recordedAudio"
              class="action-btn"
              @click="transcribeAudio"
            >
              <el-icon><Headset /></el-icon>
              Transcribe
            </el-button>
          </div>

          <div class="output-panel">
            <div class="panel-card result-card">
              <h3>Transcription Result</h3>
              
              <div v-if="isTranscribing" class="processing-state">
                <el-icon class="is-loading"><Loading /></el-icon>
                <p>Transcribing audio...</p>
                <ProgressBar :percentage="sttProgress" />
              </div>
              
              <div v-else-if="transcriptionResult" class="transcription-result">
                <div class="result-header">
                  <el-tag>{{ transcriptionResult.language }}</el-tag>
                  <span class="duration">Duration: {{ formatDuration(transcriptionResult.duration) }}</span>
                </div>
                
                <div class="transcription-text">
                  {{ transcriptionResult.text }}
                </div>
                
                <div class="result-actions">
                  <el-button :icon="CopyDocument" @click="copyText">Copy</el-button>
                  <el-button :icon="Download" @click="downloadTranscript">Download</el-button>
                </div>
              </div>
              
              <div v-else class="empty-result">
                <el-icon><Headset /></el-icon>
                <p>Upload or record audio to see transcription</p>
              </div>
            </div>
          </div>
        </div>
      </el-tab-pane>

      <!-- Text to Speech Tab -->
      <el-tab-pane label="Text to Speech" name="tts">
        <div class="content-grid">
          <div class="input-panel">
            <div class="panel-card">
              <h3>Input Text</h3>
              <el-input
                v-model="ttsText"
                type="textarea"
                :rows="6"
                placeholder="Enter text to convert to speech..."
                class="text-input"
              />
              <div class="char-count">{{ ttsText.length }} / 5000 characters</div>
            </div>

            <div class="panel-card">
              <h3>Voice Settings</h3>
              <div class="setting-row">
                <label>Voice</label>
                <el-select v-model="ttsVoice" class="full-width">
                  <el-option 
                    v-for="voice in availableVoices" 
                    :key="voice.id" 
                    :label="voice.name"
                    :value="voice.id"
                  >
                    <div class="voice-option">
                      <span>{{ voice.name }}</span>
                      <el-tag size="small">{{ voice.language }}</el-tag>
                    </div>
                  </el-option>
                </el-select>
              </div>
              <div class="setting-row">
                <label>Speed ({{ ttsSpeed }}x)</label>
                <el-slider v-model="ttsSpeed" :min="0.5" :max="2" :step="0.1" />
              </div>
              <div class="setting-row">
                <label>Pitch</label>
                <el-slider v-model="ttsPitch" :min="-10" :max="10" :step="1" />
              </div>
            </div>

            <el-button 
              type="primary" 
              size="large"
              :loading="isSynthesizing"
              :disabled="!ttsText.trim()"
              class="action-btn"
              @click="synthesizeSpeech"
            >
              <el-icon><Microphone /></el-icon>
              Generate Speech
            </el-button>
          </div>

          <div class="output-panel">
            <div class="panel-card result-card">
              <h3>Generated Audio</h3>
              
              <div v-if="isSynthesizing" class="processing-state">
                <el-icon class="is-loading"><Loading /></el-icon>
                <p>Generating speech...</p>
              </div>
              
              <div v-else-if="synthesizedAudio" class="audio-result">
                <div class="audio-player">
                  <audio :src="synthesizedAudio" controls />
                </div>
                <div class="result-actions">
                  <el-button :icon="Download" @click="downloadAudio">Download MP3</el-button>
                  <el-button :icon="Refresh" @click="regenerateSpeech">Regenerate</el-button>
                </div>
              </div>
              
              <div v-else class="empty-result">
                <el-icon><Microphone /></el-icon>
                <p>Enter text and generate to hear speech</p>
              </div>
            </div>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { 
  UploadFilled, 
  Microphone, 
  Headset, 
  Loading, 
  CopyDocument, 
  Download,
  Refresh
} from '@element-plus/icons-vue'
import { ProgressBar } from '@/components/common'
import { audioApi } from '@/api/endpoints/audio'

const activeTab = ref('stt')

// Speech to Text
const sttInputType = ref<'upload' | 'record'>('upload')
const audioFile = ref<File | null>(null)
const audioUrl = ref('')
const recordedAudio = ref<Blob | null>(null)
const isRecording = ref(false)
const recordingTime = ref('00:00')
const sttModel = ref('whisper-large')
const sttLanguage = ref('auto')
const isTranscribing = ref(false)
const sttProgress = ref(0)
const transcriptionResult = ref<{ text: string; language: string; duration: number } | null>(null)

let mediaRecorder: MediaRecorder | null = null
let recordingInterval: ReturnType<typeof setInterval> | null = null
let recordingSeconds = 0

// Text to Speech
const ttsText = ref('')
const ttsVoice = ref('default-female')
const ttsSpeed = ref(1)
const ttsPitch = ref(0)
const isSynthesizing = ref(false)
const synthesizedAudio = ref<string | null>(null)

const availableVoices = ref([
  { id: 'default-female', name: 'Default Female', language: 'EN' },
  { id: 'default-male', name: 'Default Male', language: 'EN' },
  { id: 'zh-female', name: 'Chinese Female', language: 'ZH' },
  { id: 'zh-male', name: 'Chinese Male', language: 'ZH' }
])

const handleAudioUpload = (file: any) => {
  audioFile.value = file.raw
  audioUrl.value = URL.createObjectURL(file.raw)
}

const toggleRecording = async () => {
  if (isRecording.value) {
    stopRecording()
  } else {
    await startRecording()
  }
}

const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)
    const chunks: Blob[] = []
    
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data)
    mediaRecorder.onstop = () => {
      recordedAudio.value = new Blob(chunks, { type: 'audio/webm' })
      audioUrl.value = URL.createObjectURL(recordedAudio.value)
      stream.getTracks().forEach(track => track.stop())
    }
    
    mediaRecorder.start()
    isRecording.value = true
    recordingSeconds = 0
    
    recordingInterval = setInterval(() => {
      recordingSeconds++
      const mins = Math.floor(recordingSeconds / 60)
      const secs = recordingSeconds % 60
      recordingTime.value = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }, 1000)
  } catch (e) {
    ElMessage.error('Failed to access microphone')
  }
}

const stopRecording = () => {
  if (mediaRecorder) {
    mediaRecorder.stop()
    isRecording.value = false
    if (recordingInterval) {
      clearInterval(recordingInterval)
      recordingInterval = null
    }
  }
}

const transcribeAudio = async () => {
  const file = audioFile.value || (recordedAudio.value ? new File([recordedAudio.value], 'recording.webm') : null)
  if (!file) return

  isTranscribing.value = true
  sttProgress.value = 0

  const interval = setInterval(() => {
    if (sttProgress.value < 90) sttProgress.value += 10
  }, 500)

  try {
    const result = await audioApi.transcribe(file, sttLanguage.value !== 'auto' ? sttLanguage.value : undefined, sttModel.value)
    sttProgress.value = 100
    
    transcriptionResult.value = {
      text: result.text || 'Transcription completed',
      language: result.language || sttLanguage.value,
      duration: result.duration || 0
    }
    ElMessage.success('Transcription completed!')
  } catch (e: any) {
    ElMessage.error(e.message || 'Transcription failed')
  } finally {
    clearInterval(interval)
    isTranscribing.value = false
  }
}

const synthesizeSpeech = async () => {
  if (!ttsText.value.trim()) return

  isSynthesizing.value = true

  try {
    const result = await audioApi.synthesize({
      text: ttsText.value,
      voice: ttsVoice.value,
      speed: ttsSpeed.value,
      pitch: ttsPitch.value
    })
    
    synthesizedAudio.value = result.audio_url || ''
    ElMessage.success('Speech generated!')
  } catch (e: any) {
    ElMessage.error(e.message || 'Synthesis failed')
  } finally {
    isSynthesizing.value = false
  }
}

const copyText = () => {
  if (transcriptionResult.value) {
    navigator.clipboard.writeText(transcriptionResult.value.text)
    ElMessage.success('Copied to clipboard')
  }
}

const downloadTranscript = () => {
  if (transcriptionResult.value) {
    const blob = new Blob([transcriptionResult.value.text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'transcription.txt'
    a.click()
  }
}

const downloadAudio = () => {
  if (synthesizedAudio.value) {
    const a = document.createElement('a')
    a.href = synthesizedAudio.value
    a.download = 'speech.mp3'
    a.click()
  }
}

const regenerateSpeech = () => {
  synthesizeSpeech()
}

const formatDuration = (seconds: number) => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

onUnmounted(() => {
  stopRecording()
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.audio-processing-view { padding: 24px; }

.page-header {
  margin-bottom: 32px;
  h1 { font-size: 2rem; background: $gradient-neon; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  p { color: $text-secondary; }
}

.audio-tabs {
  :deep(.el-tabs__header) { margin-bottom: 24px; }
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  @media (max-width: 1200px) { grid-template-columns: 1fr; }
}

.panel-card {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  h3 { font-size: 1rem; color: $text-primary; margin-bottom: 16px; }
}

.input-selector { width: 100%; margin-bottom: 20px; }
.full-width { width: 100%; }
.setting-row { margin-bottom: 16px; label { display: block; font-size: 13px; color: $text-secondary; margin-bottom: 8px; } }

.audio-upload { width: 100%; }

.audio-preview {
  margin-top: 16px;
  audio { width: 100%; }
}

.record-area {
  text-align: center;
  padding: 20px;
}

.recording-visualizer {
  display: flex;
  justify-content: center;
  gap: 4px;
  height: 60px;
  margin-bottom: 20px;
  
  .visualizer-bar {
    width: 4px;
    background: $glass-border;
    border-radius: 2px;
    transition: height 0.1s ease;
    align-self: center;
    height: 10px;
  }
  
  &.active .visualizer-bar {
    animation: visualize 0.5s ease infinite;
    background: $neon-cyan;
    
    @for $i from 1 through 20 {
      &:nth-child(#{$i}) {
        animation-delay: #{$i * 0.05}s;
      }
    }
  }
}

@keyframes visualize {
  0%, 100% { height: 10px; }
  50% { height: #{random(50) + 10}px; }
}

.recording-time {
  display: block;
  margin-top: 16px;
  font-family: monospace;
  font-size: 24px;
  color: $neon-cyan;
}

.action-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  .el-icon { margin-right: 8px; }
}

.result-card { min-height: 300px; }

.processing-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  .el-icon { font-size: 48px; color: $neon-cyan; margin-bottom: 16px; }
}

.empty-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  color: $text-muted;
  .el-icon { font-size: 64px; margin-bottom: 16px; opacity: 0.5; }
}

.transcription-result {
  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    .duration { font-size: 13px; color: $text-muted; }
  }
  
  .transcription-text {
    padding: 16px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    line-height: 1.8;
    color: $text-primary;
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 16px;
  }
}

.audio-result {
  .audio-player {
    margin-bottom: 16px;
    audio { width: 100%; }
  }
}

.result-actions {
  display: flex;
  gap: 12px;
}

.text-input {
  :deep(.el-textarea__inner) {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid $glass-border;
    color: $text-primary;
  }
}

.char-count {
  text-align: right;
  font-size: 12px;
  color: $text-muted;
  margin-top: 8px;
}

.voice-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}
</style>
