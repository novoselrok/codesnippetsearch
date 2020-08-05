<template>
    <div class="repository-search">
        <div class="repository-search__search-bar">
            <input
                placeholder="Enter your query"
                class="repository-search__search-bar__input"
                type="text"
                @keydown.enter="onSearch"
                v-model="query">
            <div class="repository-search__search-bar__button" @click="onSearch"><div>Search</div></div>
        </div>
        <div
            v-if="!isLoadingSearchResults"
            class="repository-search__code-documents">
            <code-document
                v-for="codeDocument in codeDocuments"
                :key="codeDocument.codeHash"
                :repository-organization="repository.organization"
                :repository-name="repository.name"
                :code-html="codeDocument.codeHtml"
                :code-hash="codeDocument.codeHash"
                :filename="codeDocument.filename"
                :url="codeDocument.url"
                :language="codeDocument.language"
                :distance="codeDocument.distance"
            />
        </div>
        <div v-else>
            <div class="repository-search__placeholder">
                <svg viewBox="0 0 703 401" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path fill="#E1E1E1" d="M2 0h701v88H2z"/>
                    <path fill="#E0E0E0"
                          d="M24 145h544v16H24zM24 113h200v16H24zM24 177h150v16H24zM24 241h200v16H24zM24 273h362v16H24zM24 305h544v16H24zM24 337h544v16H24zM24 369h544v16H24z"/>
                    <rect x="1.5" y="1.5" width="700" height="398" rx="6.5" stroke="#E1E1E1" stroke-width="3"/>
                </svg>
            </div>
        </div>
    </div>
</template>

<script>
import CodeDocument from './CodeDocument'
import { fetchEndpoint } from '../common'

export default {
    components: {
        CodeDocument,
    },
    props: {
        repository: { type: Object, required: true }
    },
    data () {
        return {
            isLoading: false,
            isLoadingSearchResults: false,
            isError: false,
            codeDocuments: [],
            query: ''
        }
    },
    methods: {
        onSearch () {
            if (this.routeQuery === this.query) {
                return
            }
            this.fetchSearchResults()
        },
        fetchSearchResults () {
            if (!this.query) {
                return
            }

            this.isLoadingSearchResults = true
            fetchEndpoint(`/api/repositories/${this.repository.organization}/${this.repository.name}/search?query=${this.query}`)
                .then(response => {
                    this.codeDocuments = response.codeDocuments
                    this.isLoadingSearchResults = false
                })
                .catch(error => {
                    this.isError = true
                    console.error(error)
                })
        }
    }
}
</script>

<style lang="less" scoped>
@keyframes flicker-animation {
  0%   { opacity: 1; }
  50%  { opacity: 0; }
  100% { opacity: 1; }
}

.repository-search {
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;

    &__placeholder {
        width: 100%;
        animation: flicker-animation 2s infinite;
    }

    &__code-documents {
        flex: 1;
        overflow: auto;
    }

    &__search-bar {
        display: flex;
        width: 100%;
        margin-bottom: 30px;

        &__input {
            margin-right: 8px;
            border: 3px solid #5F65EE;
            border-radius: 8px;
            width: 100%;
            font-size: 16px;
            padding: 8px;
            line-height: 25px;
        }

        &__button {
            background-color: #5F65EE;
            border-radius: 8px;
            color: white;
            padding: 8px 16px;
            font-size: 16px;
            display: flex;
            align-items: center;
            cursor: pointer;
        }
    }
}
</style>
