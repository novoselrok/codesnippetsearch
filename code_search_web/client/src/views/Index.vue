<template>
    <div>
        <header-element class="header">
            <div class="header__title">
                <div class="header__title__above">Query {{ nRepositories }} code repositories.</div>
                <div class="header__title__below">Using natural language.</div>
            </div>
        </header-element>
        <div class="repositories-wrapper">
            <div class="repositories">
                <div class="repositories__header">
                    <div class="repositories__header__left">Choose a repository</div>
                    <div class="repositories__header__right">
                        <div class="repositories__header__right__filter">
                            <div class="repositories__header__right__filter__icon">
                                <svg width="12" height="12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11.707 10.293l-2.54-2.54a5.015 5.015 0 10-1.414 1.414l2.54 2.54a1 1 0 001.414-1.414zM2 5a3 3 0 116 0 3 3 0 01-6 0z" fill="#828282"/></svg>
                            </div>
                            <input class="repositories__header__right__filter__input" type="text" placeholder="Filter repositories" v-model="repositoriesFilter">
                        </div>
                    </div>
                </div>
                <div class="repositories__list">
                    <div
                        v-for="repository in filteredRepositories"
                        :key="repository.id"
                        class="repositories__list__repository">
                        <div class="repositories__list__repository__languages">
                            <div
                                class="repositories__list__repository__languages__language"
                                v-for="language in repository.languages"
                                :key="language">
                                {{ language|capitalize }}
                            </div>
                        </div>
                        <div class="repositories__list__repository__name">
                            <router-link :to="{
                                name: 'RepositorySearch',
                                params: { repositoryOrganization: repository.organization, repositoryName: repository.name }
                            }">{{ repository.organization }} / {{ repository.name }}</router-link>
                        </div>
                        <div class="repositories__list__repository__description">{{ repository.description }}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { mapGetters } from 'vuex'
import Header from '../components/Header'

export default {
    name: 'Index',
    components: {
        'header-element': Header
    },
    data () {
        return {
            repositoriesFilter: ''
        }
    },
    computed: {
        ...mapGetters(['getRepositories']),
        filteredRepositories () {
            if (!this.repositoriesFilter) {
                return this.getRepositories
            }

            return this.getRepositories.filter(repository => {
                return repository.name.toLowerCase().indexOf(this.repositoriesFilter) > -1 ||
                    repository.organization.toLowerCase().indexOf(this.repositoriesFilter) > -1
            })
        }
    },
    beforeCreate () {
        this.nRepositories = window.N_REPOSITORIES
    }
}
</script>

<style lang="less" scoped>
    .header {
        &__title {
            color: white;
            text-align: center;
            margin-top: 40px;
            line-height: 70px;

            &__above {
                font-size: 50px;
            }

            &__below {
                font-size: 40px;
            }
        }
    }

    .repositories-wrapper {
        display: flex;
        align-items: center;
        flex-direction: column;
    }

    .repositories {
        display: flex;
        align-items: center;
        flex-direction: column;
        min-width: 300px;
        max-width: 600px;
        width: 100%;
        margin-top: 60px;
        background-color: white;

        &__header {
            width: 100%;
            display: flex;
            justify-content: space-between;
            color: #828282;
            font-size: 12px;

            &__left {
                display: flex;
                align-items: flex-end;
            }

            &__right__filter {
                position: relative;

                &__icon {
                    position: absolute;
                    left: 4px;
                    top: 8px;
                }

                &__input {
                    padding: 4px 4px 4px 20px;
                    border: 1px solid #BDBDBD;
                    box-sizing: border-box;
                    border-radius: 4px;
                }
            }
        }

        &__list {
            margin-top: 16px;
            width: 100%;

            &__repository {
                position: relative;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                border-radius: 8px;

                &__languages {
                    display: flex;
                    flex-direction: row;
                    font-family: 'Roboto Mono', monospace;
                    font-size: 12px;
                    line-height: 14px;
                    margin-bottom: 10px;

                    &__language {
                        margin-right: 10px;
                        padding: 2px 8px;
                        border: 1px solid black;
                        border-radius: 4px;
                    }
                }

                &__name a {
                    font-size: 18px;
                    line-height: 21px;
                    font-weight: bold;
                    color: #5F65EE;
                    text-decoration: none;

                    &:hover {
                        text-decoration: underline;
                    }
                }

                &__description {
                    margin-top: 8px;
                    font-size: 12px;
                    line-height: 14px;
                }
            }
        }
    }

    @media only screen
    and (max-device-width : 667px) {
        .header {
            &__title {
                margin-top: 40px;
                line-height: 50px;

                &__above {
                    font-size: 30px;
                }

                &__below {
                    font-size: 20px;
                }
            }
        }

        .repositories-wrapper {
            padding: 16px;
        }
    }
</style>
