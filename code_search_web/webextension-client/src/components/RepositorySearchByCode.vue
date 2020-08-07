<template>
    <div class="repository-search-by-code">
        <div class="repository-search-by-code__code">
            <textarea
                placeholder="Enter the code"
                class="repository-search-by-code__code__textarea"
                v-model="code"></textarea>
            <div class="repository-search-by-code__code__bottom">
                <div class="repository-search-by-code__code__bottom__languages">
                    <select v-model="language">
                      <option disabled value="">Select a language</option>
                      <option v-for="language in repository.languages" :key="language" :value="language">{{ language|capitalize }}</option>
                    </select>
                </div>
                <div class="repository-search-by-code__code__bottom__button" @click="onSearch"><div>Search</div></div>
            </div>
        </div>
        <div
            v-if="!isLoadingSearchResults"
            class="repository-search-by-code__code-documents">
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
            <div class="repository-search-by-code__placeholder">
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
        repository: { type: Object, required: true },
        contextMenuSearchByCode: { type: Object, required: false }
    },
    data () {
        return {
            isLoadingSearchResults: false,
            isError: false,
            codeDocuments: [],
            code: '',
            language: null,
        }
    },
    methods: {
        onSearch () {
            this.fetchSearchResults()
        },
        fetchSearchResults () {
            if (!this.code || !this.language) {
                return
            }

            this.isLoadingSearchResults = true
            fetchEndpoint(`/api/repositories/${this.repository.organization}/${this.repository.name}/searchByCode?code=${this.code}&language=${this.language}`)
                .then(response => {
                    this.codeDocuments = response.codeDocuments
                    this.isLoadingSearchResults = false
                })
                .catch(error => {
                    this.isError = true
                    console.error(error)
                })
        }
    },
    watch: {
        contextMenuSearchByCode () {
            this.code = this.contextMenuSearchByCode.code
            this.language = this.contextMenuSearchByCode.language
            this.fetchSearchResults()
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

.repository-search-by-code {
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

    &__code {
        display: flex;
        flex-direction: column;
        margin-bottom: 30px;

        &__textarea {
            font-family: monospace;
            width: 100%;
            height: 100px;
            max-height: 500px;
            border: 3px solid #5F65EE;
            border-radius: 8px;
            padding: 8px;
            resize: vertical;
        }

        &__bottom {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;

            &__languages {
                display: flex;
                align-items: center;
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
}
</style>
