import Vue from 'vue'
import App from './App.vue'
import { fetchEndpoint } from './common'

Vue.config.productionTip = false
Vue.filter('capitalize', function (value) {
    if (!value) return ''
    value = value.toString()
    return value.charAt(0).toUpperCase() + value.slice(1)
})
const extensionToLanguage = {
    'py': 'python',
    'rb': 'ruby',
    'go': 'go',
    'php': 'php',
    'java': 'java',
    'js': 'javascript',
}

const getLastElement = arr => arr.slice(-1)[0]

window.API_URL = process.env.VUE_APP_DEV_API_URL || 'https://codesnippetsearch.net'

const initApp = (repository) => {
    const sidebarEl = document.createElement('div')
    sidebarEl.classList.add('codesnippetsearch-sidebar')
    sidebarEl.innerHTML = '<div id="app"></div>'
    document.body.appendChild(sidebarEl)

    const app = new Vue({
        data () {
            return {
                contextMenuSearchByCode: {},
            }
        },
        render (h) {
            return h(App, {
                props: {
                    repository,
                    contextMenuSearchByCode: this.contextMenuSearchByCode
                }
            })
        }
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
            sidebarEl.style.display = 'block'

            const selection = window.getSelection()
            if (!selection) {
                return
            }
            let language = null
            const selectedNode = selection.anchorNode.nodeType === Node.TEXT_NODE ?
                selection.anchorNode.parentNode : selection.anchorNode
            const file = selectedNode.closest('.file')
            if (file) {
                // If in a pull request
                language = extensionToLanguage[file.dataset.fileType.slice(1)]
            } else if (window.location.pathname.indexOf('/blob') > -1) {
                // If viewing the blob
                const filename = getLastElement(window.location.pathname.split('/'))
                language = extensionToLanguage[getLastElement(filename.split('.'))]
            }
            app.contextMenuSearchByCode = {
                code: data.code,
                language
            }
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
