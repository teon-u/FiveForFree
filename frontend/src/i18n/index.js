import ko from './locales/ko.json'
import en from './locales/en.json'

const locales = {
  ko,
  en,
}

export const getTranslation = (language, key) => {
  const keys = key.split('.')
  let value = locales[language]

  for (const k of keys) {
    if (value && typeof value === 'object') {
      value = value[k]
    } else {
      return key // Return key if translation not found
    }
  }

  return value || key
}

export const t = (language) => (key) => getTranslation(language, key)

export { locales }
