import Vue from 'vue'
import Vuex from 'vuex'
import { fetchEndpoint } from '../common'

Vue.use(Vuex)

export default new Vuex.Store({
    state: {
        repositories: []
    },
    mutations: {
        SET_REPOSITORIES: (state, repositories) => {
            state.repositories = repositories
        }
    },
    actions: {
        initialize: ({ commit }) => {
            fetchEndpoint('/api/repositories')
                .then(response => commit('SET_REPOSITORIES', response.codeRepositories))
        }
    },
    getters: {
        getRepositories: state => state.repositories,
        getRepository: state => (organization, name) => state.repositories.find(
            r => r.organization === organization && r.name === name)
    },
    modules: {
        repositorySearch: {
            namespaced: true,
            state: {
                repository: {},
                searchQuery: null,
                codeDocuments: []
            }
        },
        codeDocument: {
            namespaced: true,
            state: {
                codeDocument: {}
            }
        }
    }
})
