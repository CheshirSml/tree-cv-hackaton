
<template>
  <div class="misc-wrapper">
    <ErrorHeader
      status-code="404"
      title="Страница не найдена ⚠️"
      description="Попробуйте вернуться на главную и поискать иначе."
    />

    <!-- 👉 Image -->
    <div class="misc-avatar w-100 text-center">
      <VBtn
        to="/"
        class="mt-10"
      >
        Вернуться на главную
      </VBtn>
    </div>

  </div>
</template>

<style lang="scss">
@use "@core/scss/template/pages/misc.scss";

.misc-footer-tree {
  inline-size: 15.625rem;
  inset-block-end: 3.5rem;
  inset-inline-start: 0.375rem;
}
</style>
