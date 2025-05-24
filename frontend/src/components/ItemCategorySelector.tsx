import React, { useEffect, useState, useRef } from 'react'
import axios from 'axios'

interface Props {
  selectedCategories: string[]
  selectedItems: string[]
  onCategoryChange: (categories: string[]) => void
  onItemChange: (items: string[]) => void
}

const ItemCategorySelector: React.FC<Props> = ({ selectedCategories, selectedItems, onCategoryChange, onItemChange }) => {
  const [categories, setCategories] = useState<string[]>([])
  const [items, setItems] = useState<string[]>([])
  const [lastCategoryIndex, setLastCategoryIndex] = useState<number | null>(null)
  const [lastItemIndex, setLastItemIndex] = useState<number | null>(null)

  // fetch categories on mount
  useEffect(() => {
    axios.get('/api/categories')
      .then(res => {
        const sorted = res.data.sort((a: string, b: string) => a.localeCompare(b, 'ko'));
        setCategories(sorted);
      })
      .catch(err => console.error(err))
  }, [])

  // fetch items when categories change
  useEffect(() => {
    if (selectedCategories.length > 0) {
      const qs = selectedCategories.map(encodeURIComponent).join(',')
      axios.get(`/api/items?category=${qs}`)
        .then(res => {
          const sorted = res.data.sort((a: string, b: string) => a.localeCompare(b, 'ko'));
          setItems(sorted);
        })
        .catch(err => console.error(err))
    } else {
      axios.get('/api/items')
        .then(res => {
          const sorted = res.data.sort((a: string, b: string) => a.localeCompare(b, 'ko'));
          setItems(sorted);
        })
        .catch(err => console.error(err))
    }
  }, [selectedCategories])

  // dropdown state and handlers for multi-select
  const [openCategories, setOpenCategories] = useState(false);
  const [openItems, setOpenItems] = useState(false);
  // refs for detecting outside clicks
  const catWrapperRef = useRef<HTMLDivElement>(null);
  const itemWrapperRef = useRef<HTMLDivElement>(null);

  // close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (openCategories && catWrapperRef.current && !catWrapperRef.current.contains(event.target as Node)) {
        setOpenCategories(false);
      }
      if (openItems && itemWrapperRef.current && !itemWrapperRef.current.contains(event.target as Node)) {
        setOpenItems(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => { document.removeEventListener('mousedown', handleClickOutside); };
  }, [openCategories, openItems]);

  // dynamic labels
  const categoryLabel = selectedCategories.length === 0
    ? '전체 선택'
    : selectedCategories.length === 1
      ? selectedCategories[0]
      : `${selectedCategories[0]} 외 ${selectedCategories.length - 1}개`;
  const itemLabel = selectedItems.length === 0
    ? '전체 선택'
    : selectedItems.length === 1
      ? selectedItems[0]
      : `${selectedItems[0]} 외 ${selectedItems.length - 1}개`;

  const handleAllCategories = (e: React.ChangeEvent<HTMLInputElement>) => {
    const checked = e.target.checked;
    onCategoryChange(checked ? [...categories] : []);
  };

  const handleCategoryCheckbox = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const checked = e.target.checked;
    const newSelected = checked
      ? Array.from(new Set([...selectedCategories, value]))
      : selectedCategories.filter(c => c !== value);
    onCategoryChange(newSelected);
  };

  const handleAllItems = (e: React.ChangeEvent<HTMLInputElement>) => {
    const checked = e.target.checked;
    onItemChange(checked ? [...items] : []);
  };

  const handleItemCheckbox = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const checked = e.target.checked;
    const newSelected = checked
      ? Array.from(new Set([...selectedItems, value]))
      : selectedItems.filter(i => i !== value);
    onItemChange(newSelected);
  };

  return (
    <div className="control-panel multi-selects">
      <label>분류: </label>
      <div className="multi-select-wrapper" ref={catWrapperRef}>
        <button
          type="button"
          className="multi-select-btn"
          onClick={() => setOpenCategories(o => !o)}
        >
          {categoryLabel}
        </button>
        {openCategories && (
          <div className="multi-dropdown">
            <label>
              <input
                type="checkbox"
                checked={selectedCategories.length === categories.length}
                onChange={handleAllCategories}
              />
              전체 선택
            </label>
            {categories.map(cat => (
              <label key={cat}>
                <input
                  type="checkbox"
                  value={cat}
                  checked={selectedCategories.includes(cat)}
                  onChange={handleCategoryCheckbox}
                />
                {cat}
              </label>
            ))}
          </div>
        )}
      </div>
      <label>품목: </label>
      <div className="multi-select-wrapper" ref={itemWrapperRef}>
        <button
          type="button"
          className="multi-select-btn"
          onClick={() => setOpenItems(o => !o)}
        >
          {itemLabel}
        </button>
        {openItems && (
          <div className="multi-dropdown">
            <label>
              <input
                type="checkbox"
                checked={selectedItems.length === items.length}
                onChange={handleAllItems}
              />
              전체 선택
            </label>
            {items.map(it => (
              <label key={it}>
                <input
                  type="checkbox"
                  value={it}
                  checked={selectedItems.includes(it)}
                  onChange={handleItemCheckbox}
                />
                {it}
              </label>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default ItemCategorySelector