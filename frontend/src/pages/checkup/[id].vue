<script setup lang="ts">
import treePhoto1 from '@images/trees/tree1.jpg';
import treePhoto2 from '@images/trees/tree2.jpg';
import { ref } from "vue";

interface Tree {
  id: number
  photo: string
  title: string
  status: string
  defects: string[]
  dryBranches?: string
}

const trees = ref<Tree[]>([
  {
    id: 1,
    photo: treePhoto1,
    title: "Липа мелколистная",
    status: "Здоровая",
    dryBranches: "5% сухих ветвей",
    defects: ["дупло", "сломанная ветвь"],
  },
  {
    id: 2,
    photo: treePhoto2,
    title: "Клён обыкновенный",
    status: "Аварийная",
    defects: ["трещина ствола", "наклон более 30°"],
  },
])
</script>

<template>
  <v-container class="pa-4" style="max-width: 480px">
    <!-- Заголовок участка -->
    <h2 class="text-h4 font-weight-bold mb-4">
      Участок А-12, Сокольники
    </h2>

    <!-- Обобщение -->
    <v-card class="pa-4 mb-6 rounded-lg" elevation="3">
      <div class="d-flex justify-space-around mb-4">
        <!-- Здоровые -->
        <v-chip color="success" variant="flat" class="px-3">
          <strong>3</strong>&nbsp; здоровые
        </v-chip>

        <!-- Неудовлетворительные -->
        <v-chip color="warning" variant="flat" class="px-3">
          <strong>1</strong>&nbsp; неуд.
        </v-chip>

        <!-- Аварийные -->
        <v-chip color="error" variant="flat" class="px-3">
          <strong>2</strong>&nbsp; аварийные
        </v-chip>
      </div>

      <v-divider />

      <div class="mt-4 text-center">
        <span class="text-subtitle-1 font-weight-medium text-medium-emphasis">
          Обследование <strong>6 деревьев</strong> завершено
        </span>
      </div>
    </v-card>

    <!-- Список деревьев -->
    <div v-for="tree in trees" :key="tree.id" class="mb-6">
      <v-card class="rounded-lg overflow-hidden" elevation="2">
        <!-- Фото -->
        <v-img :src="tree.photo" height="240" cover />

        <v-card-text>
          <!-- Название -->
          <h3 class="text-h5 font-weight-bold mb-2">
            {{ tree.title }}
          </h3>

          <!-- Статус -->
          <div class="mb-2">
            <strong>Статус:</strong> {{ tree.status }}
          </div>

          <!-- Сухие ветви -->
          <div v-if="tree.dryBranches" class="mb-2">
            {{ tree.dryBranches }}
          </div>

          <!-- Дефекты -->
          <ul class="pl-4">
            <li v-for="defect in tree.defects" :key="defect">
              {{ defect }}
            </li>
          </ul>
        </v-card-text>
      </v-card>
    </div>
  </v-container>
</template>
