<template>
    <div class="app">
        <div class="app__main-wrapper">
            <div class="app__main">
                <div class="app__main__title-bar">
                    <div class="app__main__title-bar__title">CodeSnippetSearch</div>
                    <div class="app__main__title-bar__close" @click="$root.$emit('hide')">
                        <svg height="24" width="24" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg"><path d="M512.001 84.853L427.148 0 256.001 171.147 84.853 0 0 84.853 171.148 256 0 427.148l84.853 84.853 171.148-171.147 171.147 171.147 84.853-84.853L340.853 256z"/></svg>
                    </div>
                </div>
                <div class="app__main__navigation">
                    <div class="app__main__navigation__tabs">
                        <div
                            class="app__main__navigation__tabs__tab"
                            :class="{ 'app__main__navigation__tabs__tab--active': searchBy === 'query' }"
                            @click="searchBy = 'query'">
                            Search by query</div>
                        <div
                            class="app__main__navigation__tabs__tab"
                            :class="{ 'app__main__navigation__tabs__tab--active': searchBy === 'code' }"
                            @click="searchBy = 'code'">
                            Search by code</div>
                    </div>
                </div>
                <div class="app__main__content">
                    <RepositorySearch v-show="searchBy === 'query'" :repository="repository"></RepositorySearch>
                    <RepositorySearchByCode v-show="searchBy === 'code'" :repository="repository" :context-menu-search-by-code="contextMenuSearchByCode"></RepositorySearchByCode>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import RepositorySearch from './components/RepositorySearch'
import RepositorySearchByCode from './components/RepositorySearchByCode'

export default {
    components: {
        RepositorySearch,
        RepositorySearchByCode
    },
    props: {
        repository: { type: Object, required: true },
        contextMenuSearchByCode: { type: Object, required: false },
    },
    data () {
        return {
            searchBy: 'query',
        }
    },
    watch: {
        contextMenuSearchByCode () {
            this.searchBy = 'code'
        }
    }
}
</script>

<style lang="less" scoped>
.app {
    height: 100%;

    &__main-wrapper {
        display: flex;
        justify-content: center;
        height: 100%;
    }

    &__main {
        max-width: 500px;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;

        &__title-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;

            &__title {
                font-size: 30px;
                margin-bottom: 15px;
            }

            &__close {
                cursor: pointer;
            }
        }

        &__navigation {
            display: flex;
            margin-bottom: 15px;

            &__tabs {
                display: flex;

                &__tab {
                    color: #5F65EE;
                    border-radius: 8px;
                    padding: 8px 16px;
                    margin-right: 8px;
                    font-size: 12px;
                    display: flex;
                    align-items: center;
                    cursor: pointer;
                }

                &__tab--active {
                    background-color: #5F65EE;
                    color: white;
                }
            }
        }

        &__content {
            flex: 1;
            overflow: hidden;
        }
    }
}
</style>
