<script lang="ts">

  import { useApi } from '@/composables/useApi';
import errors from '@/utils/errors';
import emptyProfilePhoto from '@images/avatars/empty-profile-photo.png';
import { defineComponent } from "vue";
import { toast } from 'vue3-toastify';

  const $api = useApi()

  interface DOMEvent<T extends EventTarget> extends Event {
    readonly target: T
  }

  export default defineComponent({
    props: {
      id: {
        type: Number,
        default: null
      },
      label: {
        type: String,
        default: 'Загрузить фото:'
      },
      url: {
        type: String,
        default: null
      },
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
      acceptMediaTypes: {
        type: String,
        default: '*'
      },
      defaultUrl: {
        type: String,
        default: emptyProfilePhoto
      },
      disabled: {
        type: Boolean,
        default: false
      },
    },
    data() {
      return {
        is_dragover: false,
        is_processing: false,
      }
    },
    methods: {
      // handleUpload(e : DOMEvent<HTMLInputElement>) {
      handleUpload(e : Event) {
        this.is_dragover = false
        this.is_processing = true
        if (!e.target) {
          this.is_processing = false
          alert('File not selected')
          return
        }
        const fileInput : HTMLInputElement = e.target as HTMLInputElement
        const files = fileInput.files
        if (files && files.length) {
          const selectedFile = files[0]
          const formData = new FormData()
          formData.append('photo', selectedFile)
          for(var key in this.props) {
            if (this.props[key]) {
              formData.append(key, this.props[key])
            }
          }

          var headers = {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }

          $api.put(this.api, formData, headers)
            .then((response : any) => {
              console.log('SUCCESS!!', response)
              this.is_processing = false
              this.$emit('update:id', response.data.id)
              this.$emit('update:url', response.data.preview)
            })
            .catch((error : any) => {
              console.log('FAILURE!!', error)
              this.is_processing = false
              toast.error("Не удалось загрузить фото. " + errors.getErrorText(error), {
                autoClose: 8000,
              })
            })
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
      displayPhotoUrl() {
        if (!this.url && !!this.defaultUrl) {
          return this.defaultUrl
        }
        return this.url ? this.url : ''        
      }
    }
  })
</script>

<template>
  <div class="d-flex">
    <VAvatar
      rounded="lg"
      size="100"
      class="me-6"
      :image="displayPhotoUrl"
    />
    <div class="d-flex flex-column justify-center gap-5">
      <div class="d-flex flex-wrap gap-2">
        <label>
          <input
            ref="file"
            type="file"
            name="photo"
            class="form-control"
            hidden
            :accept="acceptMediaTypes"
            @change="handleUpload"
          >
          <VBtn color="primary" @click="clickOnFile"
            :disabled="is_processing || disabled"
            :loading="is_processing"
          >
            <VIcon
              icon="ri-upload-cloud-line"
              class="d-sm-none"
            />
              <span>
                Загрузить фотографию
              </span>

          </VBtn>
        </label>

        <!-- <VBtn
          type="reset"
          color="error"
          variant="outlined"
          :disabled="is_processing || !url"
          @click="resetAvatar"
        >
          <span class="d-none d-sm-block">Очистить</span>
          <VIcon icon="ri-refresh-line" class="d-sm-none" />
        </VBtn> -->
      </div>

      <p class="text-body-1 mb-0">
        JPG, GIF or PNG (Максимум 10мб)
      </p>
    </div>

  </div>
</template>
