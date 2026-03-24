import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

/**
 * Drag-and-drop / click-to-browse image uploader.
 *
 * @param {Object}   props
 * @param {Function} props.onImageSelect  - Called with the selected File
 * @param {boolean}  props.isLoading      - Disables interaction while API call is in progress
 */
export default function Uploader({ onImageSelect, isLoading }) {
  const [preview, setPreview] = useState(null)

  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0]
      if (!file) return
      const url = URL.createObjectURL(file)
      setPreview(url)
      onImageSelect(file)
    },
    [onImageSelect]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10 MB
  })

  return (
    <div
      {...getRootProps()}
      className={[
        'relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer',
        'transition-all duration-200',
        isDragActive
          ? 'border-blue-400 bg-blue-50 scale-[1.01]'
          : 'border-gray-300 bg-gray-50 hover:border-blue-300 hover:bg-blue-50/40',
        isLoading ? 'pointer-events-none opacity-60' : '',
      ].join(' ')}
    >
      <input {...getInputProps()} />

      {preview ? (
        <img
          src={preview}
          alt="Preview"
          className="mx-auto max-h-64 rounded-xl object-contain"
        />
      ) : (
        <div className="py-8">
          <div className="text-5xl mb-4">📦</div>
          <p className="text-lg font-medium text-gray-700">
            {isDragActive ? 'Drop it here!' : 'Drag & drop a product image'}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            or click to browse · JPG, PNG, WebP · max 10 MB
          </p>
        </div>
      )}
    </div>
  )
}
