<script setup lang="ts">

const props = defineProps({
  checkupData: {
    type: Object as () => any,
    required: true
  }
})

// вычисляем количество по состоянию
const counts = computed(() => {
  const healthy = props.checkupData.photos.filter((p: any) => p.annotation.condition === 'нормальное').length
  const unsatisfactory = props.checkupData.photos.filter((p: any) => p.annotation.condition === 'не удовлетворительное').length
  const critical = props.checkupData.photos.filter((p: any) => !['нормальное', 'не удовлетворительное'].includes(p.annotation.condition)).length

  return { healthy, unsatisfactory, critical }
})

const baseUrl = computed(() => {
  return import.meta.env.VITE_BASE_URL.replace('/tapi', '').replace('/api', '')
})

</script>

<template>
  <!-- Обобщение -->
  <v-card class="pa-4 mb-6 rounded-lg" elevation="3">
    <div class="d-flex justify-space-around mb-4">
      <v-chip color="success" variant="flat" class="px-3">
        <strong>{{ counts.healthy }}</strong>&nbsp; здоровые
      </v-chip>

      <v-chip color="warning" variant="flat" class="px-3">
        <strong>{{ counts.unsatisfactory }}</strong>&nbsp; неуд.
      </v-chip>

      <v-chip color="error" variant="flat" class="px-3">
        <strong>{{ counts.critical }}</strong>&nbsp; аварийные
      </v-chip>
    </div>

    <v-divider />

    <div class="mt-4 text-center">
      <span class="text-subtitle-1 font-weight-medium text-medium-emphasis">
        Обследование <strong>{{ checkupData.photos.length }} деревьев</strong> завершено
      </span>
    </div>
  </v-card>

  <!-- Список деревьев -->
  <div v-for="tree in checkupData.photos" :key="tree.id" class="mb-6">
    <v-card v-if="tree.annotation" class="rounded-lg overflow-hidden" elevation="2">
      <!-- Фото (берём annotated если есть, иначе оригинал) -->
      <v-img :src="baseUrl + (tree.annotation.annotated_photo || tree.preview)" cover />

      <v-card-text>
        <!-- Порода дерева -->
        <h3 class="text-h5 font-weight-bold mb-2">
          {{ tree.annotation.breed }}
        </h3>

        <!-- Состояние -->
        <div class="mb-2">
          <strong>Состояние:</strong> {{ tree.annotation.condition }}
        </div>

        <!-- Сухие ветви -->
        <div v-if="tree.annotation.percentage_dried > 0" class="mb-2">
          <strong>Сухие ветви:</strong> {{ tree.annotation.percentage_dried }}%
        </div>

        <!-- Артефакты -->
        <div v-if="tree.annotation.artifacts && tree.annotation.artifacts.length" class="mb-2">
          <strong>Дефекты:</strong>
          <ul class="pl-4">
            <li v-for="artifact in tree.annotation.artifacts" :key="artifact">
              {{ artifact }}
            </li>
          </ul>
        </div>

        <!-- Краткое описание -->
        <div v-if="tree.annotation.description" class="mt-2">
          <em>{{ tree.annotation.description }}</em>
        </div>
      </v-card-text>
    </v-card>
  </div>
</template>
