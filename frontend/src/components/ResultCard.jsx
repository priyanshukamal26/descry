import ConfidenceBar from './ConfidenceBar'

const CATEGORY_ICONS = {
  shoes:       '👟',
  sneakers:    '👟',
  boots:       '🥾',
  smartphone:  '📱',
  laptop:      '💻',
  tablet:      '📱',
  watch:       '⌚',
  'cap':       '🧢',
  'hat':       '🎩',
  't-shirt':   '👕',
  jacket:      '🧥',
  dress:       '👗',
  handbag:     '👜',
  headphones:  '🎧',
  sunglasses:  '🕶️',
  backpack:    '🎒',
  jeans:       '👖',
  camera:      '📷',
  bottle:      '🍶',
  wallet:      '👛',
  belt:        '👔',
}

/**
 * Card displaying full prediction results from the API.
 *
 * @param {Object} props
 * @param {Object} props.result - API response object
 */
export default function ResultCard({ result }) {
  if (!result) return null

  const icon = CATEGORY_ICONS[result.category?.toLowerCase()] ?? '🏷️'

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      {/* ── Header ────────────────────────────────────── */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{icon}</span>
          <div>
            <h2 className="text-white font-bold text-xl leading-tight">
              {result.brand}
            </h2>
            <p className="text-slate-300 text-sm capitalize">{result.category}</p>
          </div>
          {result.logo_detected && (
            <span className="ml-auto text-xs px-2 py-1 bg-green-500/20 text-green-300 rounded-full border border-green-500/30">
              Logo detected
            </span>
          )}
        </div>
      </div>

      <div className="px-6 py-5 space-y-5">
        {/* ── Confidence scores ─────────────────────────── */}
        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Confidence Scores
          </p>
          <div className="space-y-3">
            <ConfidenceBar
              label={`Brand: ${result.brand}`}
              value={result.brand_confidence}
              color="blue"
            />
            <ConfidenceBar
              label={`Category: ${result.category}`}
              value={result.category_confidence}
              color="green"
            />
          </div>
        </div>

        {/* ── Attributes ────────────────────────────────── */}
        {result.attributes && Object.keys(result.attributes).length > 0 && (
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Attributes
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(result.attributes).map(([key, val]) => (
                <span
                  key={key}
                  className="text-xs px-3 py-1 bg-slate-100 text-slate-600 rounded-full"
                >
                  {key}: <strong>{val}</strong>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* ── Top-3 alternatives ────────────────────────── */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Top Brands
            </p>
            {result.top3_brands?.map((b, i) => (
              <div
                key={i}
                className="flex justify-between text-sm py-1 border-b border-gray-50 last:border-0"
              >
                <span className={i === 0 ? 'font-semibold text-blue-600' : 'text-gray-500'}>
                  {b.brand}
                </span>
                <span className="text-gray-400">{Math.round(b.confidence * 100)}%</span>
              </div>
            ))}
          </div>
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Top Categories
            </p>
            {result.top3_categories?.map((c, i) => (
              <div
                key={i}
                className="flex justify-between text-sm py-1 border-b border-gray-50 last:border-0"
              >
                <span className={i === 0 ? 'font-semibold text-green-600' : 'text-gray-500'}>
                  {c.category}
                </span>
                <span className="text-gray-400">{Math.round(c.confidence * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
