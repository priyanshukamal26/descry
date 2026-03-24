/**
 * Animated confidence bar with label and percentage.
 *
 * @param {Object} props
 * @param {number} props.value  - Confidence value 0.0 – 1.0
 * @param {string} props.label  - Text label shown above the bar
 * @param {'blue'|'green'|'orange'} props.color - Bar accent colour
 */
export default function ConfidenceBar({ value, label, color = 'blue' }) {
  const pct = Math.round(value * 100)

  const colorMap = {
    blue:   'bg-blue-500',
    green:  'bg-green-500',
    orange: 'bg-orange-500',
  }

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-600 font-medium">{label}</span>
        <span className="font-bold text-gray-800">{pct}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${colorMap[color] ?? 'bg-blue-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
