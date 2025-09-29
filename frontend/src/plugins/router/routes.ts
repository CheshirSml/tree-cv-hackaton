export const routes = [
  // { path: '/', redirect: '/dashboard' },
  {
    path: '/',
    component: () => import('@/layouts/default.vue'),
    children: [
      {
        path: '',
        component: () => import('@/pages/index/index.vue'),
      },
      {
        path: 'checkup/new',
        component: () => import('@/pages/checkup/new.vue'),
      },
      {
        path: 'checkup/new2',
        component: () => import('@/pages/checkup/new2.vue'),
      },
      {
        path: 'checkup/details',
        component: () => import('@/pages/checkup/details.vue'),
      },
      {
        path: 'checkup/:id',
        component: () => import('@/pages/checkup/[id].vue'),
      },

    ]
  },
]
