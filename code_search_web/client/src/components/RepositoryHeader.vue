<template>
    <header-element>
        <div class="repository-header__header-wrapper">
            <div class="repository-header__header">
                <div class="repository-header__header__languages">
                    <div
                        class="repository-header__header__languages__language"
                        v-for="language in languages"
                        :key="language">
                        {{ language|capitalize }}
                    </div>
                </div>
                <div class="repository-header__header__name">
                    {{ organization }} / {{ name }}
                </div>
                <div class="repository-header__header__description">
                    {{ description }}
                </div>
            </div>
        </div>
    </header-element>
</template>

<script>
import Header from './Header'

export default {
    name: 'RepositoryHeader',
    components: {
        'header-element': Header
    },
    props: {
        name: { type: String },
        organization: { type: String },
        description: { type: String },
        languages: { type: Array }
    }
}

</script>

<style lang="less" scoped>
    .repository-header {
        &__header-wrapper {
            display: flex;
            justify-content: center;
        }

        &__header {
            width: 600px;
            color: white;

            &__languages {
                margin-top: 80px;
                font-family: 'Roboto Mono', monospace;
                font-size: 12px;
                line-height: 14px;
                display: flex;
                flex-direction: row;

                &__language {
                    background-color: white;
                    color: black;
                    padding: 2px 8px;
                    border-radius: 4px;
                    margin-right: 10px;
                }
            }

            &__name {
                margin-top: 16px;
                line-height: 47px;
                font-size: 40px;
            }

            &__description {
                margin-top: 16px;
                font-size: 12px;
                line-height: 14px;
            }
        }
    }
</style>
