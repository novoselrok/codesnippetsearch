<template>
    <div class="code-document">
        <div class="code-document__header">
            <div class="code-document__header__file">
                <a :href="url">{{ filename }}</a>
            </div>
            <div class="code-document__header__rating">
                <template v-if="distance">Match rating: {{ rating }}% &middot; </template>
            </div>
        </div>
        <div class="code-document__code" v-html="codeHtml">
        </div>
    </div>
</template>

<script>
export default {
    name: 'CodeDocument',
    props: {
        repositoryOrganization: { type: String },
        repositoryName: { type: String },
        codeHtml: { type: String },
        filename: { type: String },
        url: { type: String },
        codeHash: { type: String },
        language: { type: String },
        distance: { type: Number }
    },
    computed: {
        rating () {
            return (((2.0 - this.distance) / 2.0) * 100).toFixed(1)
        }
    }
}
</script>

<style lang="less" scoped>
.code-document {
    border: 3px solid #5F65EE;
    border-radius: 8px;
    margin-bottom: 20px;

    &__header {
        color: white;
        background-color: #5F65EE;
        padding: 8px;

        a, a:active, a:visited {
            text-decoration: underline;
            color: white;
        }

        &__file {
            margin-bottom: 8px;
            font-size: 18px;
            font-weight: bold;
        }

        &__rating {
            font-size: 16px;
        }
    }

    &__code {
        overflow: hidden;
        border-radius: 8px;
    }
}
</style>

<style lang="less">
.codesnippetsearch-highlight {
    font-size: 16px;
    max-height: 500px;
    background-color: #fff;
    overflow: auto;
    padding: 8px;
    border-radius: 8px;
}
</style>
