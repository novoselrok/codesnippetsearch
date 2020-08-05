import Vue from 'vue'
import App from './App.vue'
import { fetchEndpoint } from './common'

Vue.config.productionTip = false
Vue.filter('capitalize', function (value) {
    if (!value) return ''
    value = value.toString()
    return value.charAt(0).toUpperCase() + value.slice(1)
})

window.API_URL = process.env.VUE_APP_DEV_API_URL || 'https://codesnippetsearch.net'

const initApp = (repository) => {
    const sidebarEl = document.createElement('div')
    sidebarEl.classList.add('codesnippetsearch-sidebar')
    sidebarEl.innerHTML = '<div id="app"></div>'
    document.body.appendChild(sidebarEl)

    const app = new Vue({
        render: h => h(App, {
            props: {
                repository
            }
        }),
    }).$mount('#app')

    app.$on('hide', () => {
        sidebarEl.style.display = 'none'
    })

    /* eslint-disable */
    browser.runtime.onMessage.addListener(data => {
        switch (data.type) {
        case 'open-sidebar':
            sidebarEl.style.display = 'block'
            break;
        case 'search-by-code':
            break;
        }
    })
    /* eslint-enable */
}

const main = () => {
    const pathName = window.location.pathname
    const repositoryUrlPart = pathName.slice(1).split('/').slice(0, 2)

    if (repositoryUrlPart.length !== 2) {
        return
    }

    const [organization, name] = repositoryUrlPart
    fetchEndpoint(`/api/repositories/${organization}/${name}`).then(initApp)
}

main()
