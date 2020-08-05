import Vue from 'vue'
import VueRouter from 'vue-router'
import Index from '../views/Index.vue'
import RepositorySearch from '../views/RepositorySearch'
import SimilarCodeDocuments from '../views/SimilarCodeDocuments'

Vue.use(VueRouter)

const routes = [
    {
        path: '/',
        name: 'Index',
        component: Index
    },
    {
        path: '/:repositoryOrganization/:repositoryName',
        name: 'RepositorySearch',
        component: RepositorySearch
    },
    {
        path: '/:repositoryOrganization/:repositoryName/:codeHash',
        name: 'SimilarCodeDocuments',
        component: SimilarCodeDocuments
    }
]

const router = new VueRouter({
    mode: 'history',
    base: process.env.BASE_URL,
    routes
})

export default router
