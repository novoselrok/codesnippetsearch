<template>
    <div class="similar-code-documents">
        <repository-header
            v-if="repository"
            :name="repository.name"
            :organization="repository.organization"
            :description="repository.description"
            :languages="repository.languages"
        ></repository-header>
        <div class="similar-code-documents__main-wrapper">
            <div class="similar-code-documents__main">
                <template v-if="isLoading">
                    Loading similar snippets...
                </template>
                <template v-else>
                    <code-document
                        :repository-name="repositoryName"
                        :repository-organization="repositoryOrganization"
                        :code-html="codeDocument.codeHtml"
                        :code-hash="codeDocument.codeHash"
                        :filename="codeDocument.filename"
                        :url="codeDocument.url"
                        :language="codeDocument.language"
                        :distance="codeDocument.distance"
                    />
                    <div class="similar-code-documents__main__similar-snippets-title">Similar code snippets</div>
                    <code-document
                        v-for="similarCodeDocument in similarCodeDocuments"
                        :key="similarCodeDocument.codeHash"
                        :repository-name="repositoryName"
                        :repository-organization="repositoryOrganization"
                        :code-html="similarCodeDocument.codeHtml"
                        :code-hash="similarCodeDocument.codeHash"
                        :filename="similarCodeDocument.filename"
                        :url="similarCodeDocument.url"
                        :language="similarCodeDocument.language"
                        :distance="similarCodeDocument.distance"
                    />
                </template>
            </div>
        </div>
    </div>
</template>

<script>
import { mapGetters } from 'vuex'
import { fetchEndpoint } from '../common'
import RepositoryHeader from '../components/RepositoryHeader'
import CodeDocument from '../components/CodeDocument'

export default {
    name: 'SimilarCodeDocuments',
    components: {
        CodeDocument,
        RepositoryHeader
    },
    data () {
        return {
            codeDocument: null,
            similarCodeDocuments: [],
            isLoading: false
        }
    },
    computed: {
        ...mapGetters(['getRepository']),
        codeHash () {
            return this.$route.params.codeHash
        },
        repositoryOrganization () {
            return this.$route.params.repositoryOrganization
        },
        repositoryName () {
            return this.$route.params.repositoryName
        },
        repository () {
            return this.getRepository(this.repositoryOrganization, this.repositoryName)
        }
    },
    watch: {
        codeHash () {
            this.fetchSimilarCodeDocuments()
        }
    },
    created () {
        this.fetchSimilarCodeDocuments()
    },
    methods: {
        fetchSimilarCodeDocuments () {
            this.isLoading = true
            const codeDocumentPromise = fetchEndpoint(
                `/api/codeDocument/${this.repositoryOrganization}/${this.repositoryName}/${this.codeHash}`)
            const similarCodeDocumentsPromise = fetchEndpoint(
                `/api/similarCodeDocuments/${this.repositoryOrganization}/${this.repositoryName}/${this.codeHash}`)

            Promise.all([codeDocumentPromise, similarCodeDocumentsPromise])
                .then(responses => {
                    const [codeDocumentResponse, similarCodeDocumentsResponse] = responses
                    this.codeDocument = codeDocumentResponse.codeDocument
                    this.similarCodeDocuments = similarCodeDocumentsResponse.codeDocuments
                    this.isLoading = false
                })
        }
    }
}
</script>

<style lang="less">
    .similar-code-documents {
        &__placeholder {
            width: 100%;
            animation: flicker-animation 2s infinite;
        }

        &__main-wrapper {
            display: flex;
            justify-content: center;
        }

        &__main {
            margin-top: 40px;
            min-width: 300px;
            max-width: 600px;
            width: 100%;

            &__similar-snippets-title {
                margin-top: 40px;
                margin-bottom: 20px;
                font-size: 20px ;
            }
        }
    }
</style>
