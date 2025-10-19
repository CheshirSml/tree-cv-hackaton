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

const baseUrl = computed(() => {
  return import.meta.env.VITE_BASE_URL.replace('/tapi', '').replace('/api', '')
})

const updateCoords = (photo: any) => {
  if (!photo.coords && photo.id) {
    photo.coords = getRandomPointInSquare(props.checkupData.area_detail.coords)
    $api.patch('/photos/' + photo.id + '/update-coords/', {
      coords: photo.coords
    })
  }
}

const getRandomPointInSquare = (coords: number[]) => {
  // coords = [northLat, westLng, southLat, eastLng]
  const [northLat, westLng, southLat, eastLng] = coords;

  // Генерируем случайные координаты внутри квадрата
  const randomLat = Math.random() * (northLat - southLat) + southLat;
  const randomLng = Math.random() * (eastLng - westLng) + westLng;

  return [randomLat, randomLng];
}

</script>

<template>
  <!-- Верхние кнопки -->
  <div class="d-flex flex-column gap-3 mb-6">

    <UploadPhotosMany label="Загрузить фотографии" :api="'/photos/'" :props="{ checkup: checkupData.id }"
      accept-media-types="image/png, image/jpeg, image/gif" @uploaded="updatePhotos($event); updateCoords($event)"
      :disabled="isProcessing" />

    <v-row dense>
      <v-col v-for="photoItem, index in checkupData.photos" :key="index" cols="6" class="mb-4">
        <PhotoPreview photo-status="Готово" :photo-url="baseUrl + photoItem.preview" />
      </v-col>
    </v-row>

  </div>

  <!-- Кнопка завершить -->
  <v-btn class="mt-6" block color="success" rounded="xl" @click="finishSurvey" :loading="isProcessing"
    :disabled="isProcessing">
    Завершить обследование
  </v-btn>
</template>
