import { createModel } from '@rematch/core'
import { RootModel } from '.'

const config = window.bridge.getConfig() || {}

export const win = createModel<RootModel>()({
  state: {
    resizable: window.bridge.isWinResizable(),
    showTool: config.showTool ?? true,
    language: config.language ?? 'zh',
    ttsClone: config.ttsClone ?? false,
  } as { resizable: boolean; showTool: boolean; language: 'zh' | 'en'; ttsClone: boolean },
  reducers: {
    setResizable: (state, resizable: boolean) => {
      window.bridge.setWinResizable(resizable)
      return { ...state, resizable }
    },
    setSwitchTool: (state, showTool: boolean) => {
      return { ...state, showTool }
    },
    setLanguage: (state, language: 'zh' | 'en') => {
      return { ...state, language }
    },
    setTTSClone: (state, ttsClone: boolean) => {
      return { ...state, ttsClone }
    },
  },
})
