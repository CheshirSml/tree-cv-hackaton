<script lang="ts" setup>

  import { useApi } from '@/composables/useApi';
  const $api = useApi()

  const $props = defineProps({
    modelValue: {
      type: Number as () => number|null,
      default: null
    },
    apiUrl: {
      type: String,
      required: true
    },
    propValue: {
      type: String,
      default: 'id'
    },
    propTitle: {
      type: String,
      default: 'title'
    },
    propDefault: {
      type: String,
      default: ''
    },
    disabled: {
      type: Boolean,
      default: false
    },
    floatViewOnly: {
      type: Boolean,
      default: false
    },
    allowEmpty: {
      type: Boolean,
      default: false
    },
    required: {
      type: Boolean,
      default: false
    },
    prefixText: {
      type: String,
      default: ''
    },
    mode: {
      type: String,
      default: 'default'
    },
    className: {
      type: String,
      default: ''
    },
    label: {
      type: String,
      default: ''
    },
    excludedIds: {
      type: Array,
      default: []
    },
    emptyLabel: {
      type: String,
      default: 'Выбрать значение'
    }
  })

  const $emit = defineEmits()
  const items = ref<Array<any>|null>(null)
  const localValue = ref<number|null>(null)

  const isOpen = ref(false)


  const loadData = () => {
    $api.get($props.apiUrl)
      .then((response) => {
        items.value = response.data.filter((x: any) => !$props.excludedIds.includes(x[$props.propValue]))

        if($props.propDefault && !localValue.value) {
          const item = items.value?.find(x => x[$props.propDefault])
          if (item) {
            $emit('update:modelValue', item[$props.propValue])
          }
        }
      })
  }

  const displayValue = computed(() => {
    const item = items.value?.find((x: any) => x[$props.propValue] == localValue.value)
    return item ? item[$props.propTitle] : ''
  })

  onMounted(() => {
    document.addEventListener('click', (e: any) => {
      const target = e.target as HTMLElement
      if (!target.closest('.topbar-dropdown dropdown')) {
        isOpen.value = false
      }
    })

    localValue.value = $props.modelValue
    loadData()
  })

  watch(() => $props.apiUrl, () => loadData())
  watch(() => $props.modelValue, () => localValue.value = $props.modelValue)

</script>

<template>
  <div v-if="mode == 'float-view'" class="text-center">
    <v-menu
      open-on-click
    >
      <template v-slot:activator="{ props }">
        <v-btn
          color="primary"
          v-bind="props"
          variant="text"
          :loading="!items"
        >
          {{ displayValue ? displayValue : emptyLabel }}
          <VIcon
            icon="ri-arrow-down-s-line"
            class="ms-1 mt-1"
          />
        </v-btn>
      </template>

      <v-list>
        <v-list-item
          v-for="(item, index) in items ?? []"
          :key="index"
          link
          @click="localValue=item.id; $emit('update:modelValue', localValue)"
        >
          <v-list-item-title>{{ item[propTitle] }}</v-list-item-title>
        </v-list-item>
      </v-list>
    </v-menu>
  </div>
  <!-- <span v-if="mode == 'float-view'" :class="className">{{ displayValue }}</span>
  <div v-if="mode == 'topbar-dropdown'" class="topbar-dropdown dropdown" :class="className">
    <button v-if="items"
      class="topbar-dropdown__btn"
      type="button"
      style="margin-left: -10px;"
      @click.stop="isOpen = !isOpen"
    >
      {{ prefixText }}<span class="topbar__item-value">{{ displayValue ? displayValue : 'Выбрать значение' }}</span>
      &nbsp;
    </button>
    <div class="topbar-dropdown__body" :class="{ 'active': isOpen }">
      <div class="menu menu--layout--topbar ">
        <div class="menu__submenus-container"></div>
        <ul v-if="items" class="menu__list">
          <li v-for="item in items" class="menu__item" :class="{ 'active': item[propValue] == localValue }">
            <div class="menu__item-submenu-offset"></div>
            <a @click="$emit('update:modelValue', item[propValue])" class="menu__item-link">
              {{ item[propTitle] }}
            </a>
          </li>
        </ul>
      </div>
    </div>
  </div> -->

  <VSelect v-else
    v-model="localValue"
    :label="label"
    :items="items ?? []"
    @update:model-value="$emit('update:modelValue', localValue)"
    :class="className"
    :disabled="!items || disabled"
    :item-title="propTitle"
    :item-value="propValue"
    :loading="!items"
  />
  <!-- <select v-if="mode == 'default'"
    v-model="localValue"
    class="form-control form-control-select2"
    :disabled="!items || disabled"
    @change="$emit('update:modelValue', localValue)"
    :required="required"
    :class="className"
  >
    <option v-if="floatViewOnly" :value="null" :selected="!localValue">Выберите значение...</option>
    <option v-for="item in items" :value="item[propValue]">{{ item[propTitle] }}</option>
  </select> -->
</template>

<style>

.topbar-dropdown__body.active {
  visibility: visible;
  opacity: 1;
  transform: none;
}

</style>
