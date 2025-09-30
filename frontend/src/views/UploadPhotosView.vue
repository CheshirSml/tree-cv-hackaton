<script setup lang="ts">
import UploadPhotosMany from '@/components/common/UploadPhotosMany.vue';
import { useApi } from '@/composables/useApi';
import { toast } from 'vue3-toastify';

interface Photo {
  id: number
  src: string
  status: "В обработке" | "Ожидание" | "Готово"
}

const props = defineProps({
  checkupData: {
    type: Object,
    required: true
  }
})

const $emit = defineEmits()
const $api = useApi()
const isProcessing = ref(false)

const takePhoto = () => {
  console.log("📷 Сфотографировать дерево/куст")
}

// const uploadPhotos = () => {
//   console.log("📂 Загрузить фотографии")
// }

const finishSurvey = () => {
  isProcessing.value = true
  $api.post('/checkups/' + props.checkupData.id + '/finish/')
    .then((response) => {
      $emit('update:checkupData', response.data)
      isProcessing.value = false
      toast.success('Обследование завершено')
    })
    .catch(() => {
      isProcessing.value = false
      toast.error('Не удалось завершить обследование')
    })
}

const updatePhotos = (photo: any) => {
  const data = JSON.parse(JSON.stringify((props.checkupData)))
  data.photos.push(photo)
  $emit('update:checkupData', data)
}

</script>

<template>
  <!-- Верхние кнопки -->
  <div class="d-flex flex-column gap-3 mb-6">
    <!-- <v-btn block color="primary" rounded="xl" prepend-icon="ri-camera-line" @click="takePhoto">
                      Сфотографировать дерево/куст
                    </v-btn> -->

    <UploadPhotosMany label="Загрузить фотографии" :api="'/photos/'" :props="{ checkup: checkupData.id }"
      accept-media-types="image/png, image/jpeg, image/gif" @uploaded="updatePhotos" :disabled="isProcessing" />

    <v-row dense>
      <v-col v-for="photoItem, index in checkupData.photos" :key="index" cols="6" class="mb-4">
        <PhotoPreview photo-status="Готово" :photo-url="photoItem.preview" />
      </v-col>
    </v-row>

  </div>

  <!-- Кнопка завершить -->
  <v-btn class="mt-6" block color="success" rounded="xl" @click="finishSurvey" :disabled="isProcessing">
    Завершить обследование
  </v-btn>
</template>
