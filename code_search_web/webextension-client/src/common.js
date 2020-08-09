/* eslint-disable */
export const fetchEndpoint = endpoint => browser.runtime.sendMessage({ type: 'fetch-endpoint', endpoint })
export const postEndpoint = (endpoint, data) => browser.runtime.sendMessage({ type: 'post-endpoint', endpoint, data })
/* eslint-enable */
