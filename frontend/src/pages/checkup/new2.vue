<script setup lang="ts">
import treePhoto1 from '@images/trees/tree1.jpg';
import treePhoto2 from '@images/trees/tree2.jpg';
import treePhoto3 from '@images/trees/tree3.jpg';
import { ref } from "vue";

interface Photo {
  id: number
  src: string
  status: "В обработке" | "Ожидание" | "Готово"
}

const photos = ref<Photo[]>([
  { id: 1, src: treePhoto1, status: "В обработке" },
  { id: 2, src: treePhoto2, status: "Ожидание" },
  { id: 3, src: treePhoto3, status: "Готово" },
])

const takePhoto = () => {
  console.log("📷 Сфотографировать дерево/куст")
}

const uploadPhotos = () => {
  console.log("📂 Загрузить фотографии")
}

const finishSurvey = () => {
  console.log("✅ Завершить обследование")
}
</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">
    <!-- Верхние кнопки -->
    <div class="d-flex flex-column gap-3 mb-6">
      <v-btn block color="primary" rounded="xl" prepend-icon="ri-camera-line" @click="takePhoto">
        Сфотографировать дерево/куст
      </v-btn>

      <v-btn block color="secondary" rounded="xl" prepend-icon="ri-upload-2-line" @click="uploadPhotos">
        Загрузить фотографии
      </v-btn>
    </div>

    <!-- Фотографии -->
    <v-row dense>
      <v-col v-for="photo in photos" :key="photo.id" cols="6" class="mb-4">
        <v-sheet class="pa-2 d-flex flex-column align-center" rounded="lg" elevation="2">
          <div class="position-relative w-100">
            <v-img :src="photo.src" aspect-ratio="1" class="rounded-lg" cover />
            <!-- Лоадер поверх картинки -->
            <v-progress-circular v-if="photo.status !== 'Готово'" indeterminate color="primary" size="32"
              class="position-absolute top-50 start-50 translate-middle" />
          </div>

          <!-- Статус -->
          <div class="d-flex align-center mt-2">
            <span class="me-2 rounded-circle" :style="{
              width: '10px',
              height: '10px',
              backgroundColor:
                photo.status === 'Готово'
                  ? '#4CAF50'
                  : photo.status === 'Ожидание'
                    ? '#FFC107'
                    : '#2196F3'
            }" />
            <span class="text-caption text-medium-emphasis">
              {{ photo.status }}
            </span>
          </div>
        </v-sheet>
      </v-col>
    </v-row>

    <!-- Кнопка завершить -->
    <v-btn class="mt-6" block color="success" rounded="xl" @click="finishSurvey">
      Завершить обследование
    </v-btn>
  </v-container>
</template>
