<script lang="ts">

import { useApi } from '@/composables/useApi';
import errors from '@/utils/errors';
import { defineComponent } from "vue";
import { toast } from 'vue3-toastify';

const $api = useApi()

interface DOMEvent<T extends EventTarget> extends Event {
  readonly target: T
}

export default defineComponent({
  props: {
    // ids: {
    //   type: Array as () => Array<number>,
    //   default: null
    // },
    label: {
      type: String,
      default: 'Загрузить фотографии:'
    },
    // urls: {
    //   type: Array as () => Array<string>,
    //   default: null
    // },
    api: {
      type: String,
      required: true
    },
    props: {
      type: Object,
      required: true
    },
    isContain: {
      type: Boolean,
      default: false
    },
    disabled: {
      type: Boolean,
      default: false
    },
    acceptMediaTypes: {
      type: String,
      default: '*'
    },
    defaultUrl: {
      type: String,
      default: ''
    },
  },
  data() {
    return {
      is_dragover: false,
      is_processing: false,
      processingPhotoUrl: null as string | null,
    }
  },
  methods: {
    // handleUpload(e : DOMEvent<HTMLInputElement>) {
    async handleUpload(e: Event) {
      try {
        this.is_dragover = false
        this.is_processing = true
        if (!e.target) {
          this.is_processing = false
          alert('File not selected')
          return
        }
        const fileInput: HTMLInputElement = e.target as HTMLInputElement
        const files = fileInput.files
        if (files && files.length) {
          const selectedFile = files[0]
          const formData = new FormData()
          formData.append('photo', selectedFile)
          for (var key in this.props) {
            if (this.props[key]) {
              formData.append(key, this.props[key])
            }
          }

          var headers = {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }

          // const uploadedIds = [] as Array<number>
          // const uploadedUrls = [] as Array<string>

          await $api.post(this.api, formData, headers)
            .then((response: any) => {
              console.log('SUCCESS!!', response)
              // uploadedIds.push(response.data.id)
              // uploadedUrls.push(response.data.preview)
              this.$emit('uploaded', response.data)
            })
            .catch((error: any) => {
              console.log('FAILURE!!', error)
              toast.error("Не удалось загрузить фото. " + errors.getErrorText(error), {
                autoClose: 8000,
              })
            })
          // this.$emit('update:ids', [...this.ids, ...uploadedIds])
          // this.$emit('update:urls', [...this.urls ?? [], ...uploadedUrls])
          this.is_processing = false
        }
      }
      catch (e) {
        alert(e)
      }
    },
    clickOnFile() {
      const fileInput = this.$refs.file as HTMLElement
      fileInput.click()
    },
    resetAvatar() {
      this.$emit('update:id', null)
      this.$emit('update:url', '')
    }
  },
  computed: {
    fileElement(): HTMLInputElement | null {
      const file = this.$refs.file as HTMLInputElement | null
      return file
    },
    // displayPhotoUrls() {
    //   // if (!this.urls && !!this.defaultUrl) {
    //   //   return this.defaultUrl
    //   // }
    //   return this.urls ? this.urls : []
    // }
  }
})
</script>

<template>
  <label>
    <VBtn @click="clickOnFile" :disabled="is_processing || disabled" :loading="is_processing" class="mr-4" block
      color="secondary" rounded="xl" prepend-icon="ri-upload-2-line">
      <span style="text-transform: none;">
        {{ label }}
      </span>
    </VBtn>
    <input ref="file" type="file" name="photo" class="form-control mt-2" hidden :accept="acceptMediaTypes"
      :disabled="disabled" @change="handleUpload">
  </label>
</template>

<style>
.upload-photo-aws .is-contain {
  width: 100% !important;
}

.upload-photo-aws .is-contain img {
  object-fit: contain !important;
}

.input-file {
  position: relative;
  display: inline-block;
}

.input-file-btn {
  position: relative;
  display: inline-block;
  cursor: pointer;
  outline: none;
  text-decoration: none;
  font-size: 14px;
  vertical-align: middle;
  color: rgb(255 255 255);
  text-align: center;
  border-radius: 4px;
  background-color: #3366cc;
  line-height: 22px;
  height: 40px;
  padding: 10px 20px;
  box-sizing: border-box;
  border: none;
  margin: 0;
  transition: background-color 0.2s;
}

.input-file-text {
  padding: 0 10px;
  line-height: 40px;
  display: inline-block;
}

.input-file input[type=file] {
  position: absolute;
  z-index: -1;
  opacity: 0;
  display: block;
  width: 0;
  height: 0;
}

/* Focus */
.input-file input[type=file]:focus+.input-file-btn {
  box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, .25);
}

/* Hover/active */
.input-file:hover .input-file-btn {
  background-color: #4376dc;
}

.input-file:active .input-file-btn {
  background-color: #2356bc;
}

/* Disabled */
.input-file input[type=file]:disabled+.input-file-btn {
  background-color: #eee;
}
</style>
