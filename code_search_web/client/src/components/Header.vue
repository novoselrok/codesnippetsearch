<template>
    <div class="header">
        <div class="header__more-info">Find more information on the
            <a class="header__more-info__github" href="https://github.com/novoselrok/codesnippetsearch">GitHub</a> page</div>
        <div class="header__logo">
            <router-link :to="{ name: 'Index' }">
                <img src="/static/code_search_app/img/logo.svg" alt="Logo">
            </router-link>
        </div>
        <slot></slot>
    </div>
</template>

<script>
export default {
    name: 'Header'
}
</script>

<style lang="less" scoped>
    .header {
        display: flex;
        padding: 16px 16px 32px 16px;
        background-color: #5F65EE;
        flex-direction: column;

        &__more-info {
            color: white;
            align-self: flex-end;
            font-size: 12px;
            line-height: 14px;

            &__github, &__github:active, &__github:visited {
                color: white;
            }
        }

        &__logo {
            display: flex;
            justify-content: center;
            margin-top: 60px;
        }
    }
</style>
