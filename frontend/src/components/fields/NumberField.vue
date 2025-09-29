<script lang="ts" setup>
import { vMaska } from "maska/vue";

type ValidationRule = (value: any) => string | boolean

const $emit = defineEmits()

const $props = defineProps({
  modelValue: {
    type: Number as () => number|null,
    required: true
  },
  label: {
    type: String,
    default: ''
  },
  rules: {
    type: Array as () => ValidationRule[],
    default: () => []
  },
  disabled: {
    type: Boolean,
    default: false
  },
  hint: {
    type: String as () => string|undefined,
    default: undefined
  },
  persistentHint: {
    type: Boolean,
    default: false
  },
})

const localValue = ref<number|null>()

onMounted(() => {
  localValue.value = $props.modelValue
})

watch(() => $props.modelValue, () => {
  if (localValue.value != $props.modelValue) {
    localValue.value = $props.modelValue
  }
})

watch(() => localValue.value, () => {
  if (localValue.value != $props.modelValue) {
    $emit('update:modelValue', localValue.value)
  }
})
</script>

<template>
  <VTextField
    v-model="localValue"
    :label="label"
    :rules="rules"
    :disabled="disabled"
    :hint="hint"
    :persistent-hint="persistentHint"
    type="number"
  />
  <input type="hidden"
    v-maska="'0.99'"
    data-maska-tokens="0:\d:multiple|9:\d:optional"
    v-model="localValue"
  >
</template>
