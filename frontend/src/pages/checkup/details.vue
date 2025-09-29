<script setup lang="ts">
import treePhoto1 from '@images/trees/tree1.jpg';
import { ref } from "vue";

interface TreeDetail {
  title: string
  status: string
  dryBranches: string
  photo: string
  tags: string[]
  recommendations: string[]
}

const tree = ref<TreeDetail>({
  title: "Дуб черешчатый",
  status: "Критическое состояние",
  dryBranches: "45% сухих ветвей",
  photo: treePhoto1,
  tags: ["Дупло в стволе", "Стволовая гниль", "Сломанные ветви"],
  recommendations: [
    "Требуется санитарная обрезка",
    "Обработка дупла",
    "Возможно удаление при ухудшении состояния"
  ]
})
</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">

    <!-- Заголовок и статус -->
    <!-- <v-card class="pa-4 mb-6 rounded-lg" elevation="3"> -->
    <h2 class="text-h4 font-weight-bold mb-2">{{ tree.title }}</h2>
    <v-chip size="small" color="error" class="mb-4" variant="tonal">
      {{ tree.status }}
    </v-chip>
    <!-- <div class="text-body-2 text-medium-emphasis">
                    {{ tree.dryBranches }}
                  </div> -->
    <!-- </v-card> -->

    <!-- Фото -->
    <v-card class="mb-6 rounded-lg overflow-hidden" elevation="2">
      <v-img :src="tree.photo" cover />
    </v-card>

    <!-- Пометки (Tags) -->
    <v-card class="pa-4 mb-6 rounded-lg" elevation="2">
      <h3 class="text-subtitle-1 font-weight-medium mb-2">Пометки</h3>
      <div class="d-flex flex-wrap gap-2">
        <v-chip v-for="tag in tree.tags" :key="tag" color="warning" variant="tonal">
          {{ tag }}
        </v-chip>
      </div>
    </v-card>

    <!-- Рекомендации -->
    <v-card class="pa-4 mb-6 rounded-lg" elevation="2">
      <h3 class="text-subtitle-1 font-weight-medium mb-2">Рекомендации</h3>
      <ul class="pl-4">
        <li v-for="rec in tree.recommendations" :key="rec">
          {{ rec }}
        </li>
      </ul>
    </v-card>

  </v-container>
</template>
