import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import styled from 'styled-components'

import { debounce } from '@src/renderer/src/utils'
import LegacyRender from './Legacy'
import CurrentRender from './Current'
import Toolbar from './Toolbar'
import Tips, { TipsType } from './Tips'
import { useDispatch, useSelector } from 'react-redux'
import zhTips from './tips/zh.json'
import enTips from './tips/en.json'
import { Dispatch, RootState } from '../../store'

interface ITips {
  mouseover: Mouseover[]
  click: Mouseover[]
}

interface Season {
  date: string
  text: string
}

interface Mouseover {
  selector: string
  text: string[]
}

const Wrapper = styled.div<{ border: boolean }>`
  ${(props) => (props.border ? 'border: 2px dashed #ccc;' : 'padding: 2px;')}
  height: 100vh;
  width: 100vw;
  overflow: hidden;
`

const RenderWrapper = styled.div`
  margin-top: 20px;
`

const getCavSize = () => {
  return {
    width: window.innerWidth,
    height: window.innerHeight - 20,
  }
}

const Model = () => {
  const {
    modelPath: originModelPath,
    resizable,
    useGhProxy,
    language,
    showTool,
  } = useSelector((state: RootState) => ({
    ...state.config,
    ...state.win,
  }))

  const modelPath =
    useGhProxy && originModelPath.startsWith('http')
      ? `https://ghproxy.com/${originModelPath}`
      : originModelPath

  const dispatch = useDispatch<Dispatch>()

  const [tips, setTips] = useState<TipsType>({
    text: '',
    priority: -1,
    timeout: 0,
  })

  const [cavSize, setCavSize] =
    useState<{ width: number; height: number }>(getCavSize)

  useEffect(() => {
    ;(window as any).setSwitchTool = dispatch.win.setSwitchTool
    ;(window as any).setLanguage = dispatch.win.setLanguage
    ;(window as any).nextModel = dispatch.config.nextModel
    ;(window as any).prevModel = dispatch.config.prevModel
  }, [])

  useEffect(() => {
    const handleDragOver = (evt: DragEvent): void => {
      evt.preventDefault()
    }
    const handleDrop = async (evt: DragEvent) => {
      evt.preventDefault()

      const files = evt.dataTransfer?.files

      if (!files) {
        return
      }

      const paths = []
      for (let i = 0; i < files.length; i++) {
        const result = await window.bridge.getModels(files[i])
        paths.push(...result)
      }

      console.log('modelList: ', paths)

      if (paths.length > 0) {
        const models = paths.map((p) => `file://${p}`)

        dispatch.config.setModelList(models)
        dispatch.config.setModelPath(models[0])
      }
    }

    document.body.addEventListener('dragover', handleDragOver)
    document.body.addEventListener('drop', handleDrop)

    return () => {
      document.body.removeEventListener('dragover', handleDragOver)
      document.body.removeEventListener('drop', handleDrop)
    }
  }, [])

  useLayoutEffect(() => {
    const resizeCanvas = debounce(() => {
      setCavSize(getCavSize())
    })

    window.addEventListener('resize', resizeCanvas, false)

    return () => {
      window.removeEventListener('resize', resizeCanvas)
    }
  }, [])

  useEffect(() => {
    const handleBlur = () => {
      if (resizable) {
        dispatch.win.setResizable(false)
      }
    }

    window.addEventListener('blur', handleBlur)
    return () => {
      window.removeEventListener('blur', handleBlur)
    }
  }, [])

  const isMoc3 = modelPath.endsWith('.model3.json')

  const Render = isMoc3 ? CurrentRender : LegacyRender

  const handleMessageChange = (nextTips: TipsType) => {
    setTips(nextTips)
  }

  const handleMouseOver: React.MouseEventHandler<HTMLDivElement> = (event) => {
    const tips = tipJSONs.mouseover.find((item) =>
      (event.target as any).matches(item.selector),
    )

    if (!tips) {
      return
    }

    let text = Array.isArray(tips.text)
      ? tips.text[Math.floor(Math.random() * tips.text.length)]
      : tips.text
    text = text.replace('{text}', (event.target as HTMLDivElement).innerText)
    handleMessageChange({
      text,
      timeout: 4000,
      priority: 8,
    })
  }

// 添加一个状态来跟踪是否正在录音
const [isRecording, setIsRecording] = useState(false);

const handleClick: React.MouseEventHandler<HTMLDivElement> = async (event) => {
  let text = "思考中...";
  
  if (!isRecording) {
    // 第一次点击：开始录音
    try {
      await fetch('http://127.0.0.1:8081/start_recording', {
        method: 'POST',
      });
      setIsRecording(true);
      text = "正在录音...";
    } catch (error) {
      console.error('开始录音失败:', error);
      text = "录音启动失败";
    }
  } else {
    // 第二次点击：停止录音并获取转录结果
    try {
      const response = await fetch('http://127.0.0.1:8081/stop_recording', {
        method: 'POST',
      });
      const data = await response.json();
      text = data.transcription || "未能识别语音";
      setIsRecording(false);
    } catch (error) {
      console.error('停止录音失败:', error);
      text = "录音停止失败";
      setIsRecording(false);
    }
  }

  handleMessageChange({
    text,
    timeout: 4000,
    priority: 8,
  });
}

  const tipJSONs = language === 'en' ? enTips : zhTips

  return (
    <Wrapper
      border={resizable}
      onMouseOver={isMoc3 ? undefined : handleMouseOver}
      onClick={isMoc3 ? undefined : handleClick}
    >
      <Tips {...tips}></Tips>
      {showTool && <Toolbar onShowMessage={handleMessageChange}></Toolbar>}
      <RenderWrapper>
        <Render {...cavSize} modelPath={modelPath}></Render>
      </RenderWrapper>
    </Wrapper>
  )
}

export default Model
