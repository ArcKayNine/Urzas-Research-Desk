document.addEventListener('DOMContentLoaded', function() {
  console.log('DOMContentLoaded running')

  let hoverTimer;
  let leaveTimer;

  class HoverCard extends HTMLElement {
    connectedCallback() {
        if (!this.dataset.initialized) {
            this.dataset.initialized = "true";
            this.style.cursor = "pointer";

            this.addEventListener('mouseenter', function(e) {
              clearTimeout(hoverTimer);
              clearTimeout(leaveTimer);  // Cancel any pending hide action
              if (!isMobileDevice()) {
                showCard(this, e.clientX, e.clientY);
              }
            });

            this.addEventListener('mouseleave', function() {
              clearTimeout(hoverTimer);  // Cancel any pending hide action
              if (!isMobileDevice()) {
                hideTooltip();
              }
            });
        
        
        // // Mobile touch events
        // this.addEventListener('touchstart', function(e) {
        //   // Prevent this from firing mouseenter
        //   e.preventDefault();
          
        //   // Hide any other active tooltips first
        //   if (activeCardElement && activeCardElement !== this) {
        //     hideTooltip();
        //   }
          
        //   showCard(this, 0, 0);
        // });
        
        // // Prevent regular click behavior on mobile
        // this.addEventListener('click', function(e) {
        //   if (isMobileDevice()) {
        //     e.preventDefault();
        //   }
        // });
        };
    };
  };

  customElements.define('hover-card', HoverCard);

  // Create a cache for storing fetched card data
  const cardCache = new Map();

  // Track active tooltip for mobile toggle functionality
  let activeCardElement = null;

  // Create tooltip element that will be reused
  const tooltip = document.createElement('div');
  tooltip.className = 'mtg-tooltip';
  tooltip.style.cssText = `
    position: fixed;
    display: none;
    z-index: 1000;
    background-color: transparent;
    transition: opacity 0.2s ease-in-out;
    opacity: 0;
    background-size: contain;
    background-repeat: no-repeat;
    width: 200px;
    height: 280px;
  `;
  document.body.appendChild(tooltip);

  // Add close button for mobile
  const closeButton = document.createElement('div');
  closeButton.className = 'mtg-tooltip-close';
  closeButton.innerHTML = '×';
  closeButton.style.cssText = `
    position: absolute;
    top: -10px;
    right: -10px;
    width: 24px;
    height: 24px;
    background-color: #333;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 22px;
    font-size: 18px;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    display: none;
  `;
  tooltip.appendChild(closeButton);

  // Add styles for tooltip
  const style = document.createElement('style');
  style.textContent = `
    .mtg-tooltip {
      max-width: 223px; /* Standard card width */
      box-shadow: 0 4px 8px rgba(0,0,0,0.3);
      border-radius: 10px;
    }
    .mtg-tooltip img {
      width: 100%;
      height: auto;
      border-radius: 10px;
    }
    .mtg-tooltip.loading {
      background-color: #f5f5f5;
      width: 223px;
      height: 310px;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .mtg-tooltip .loading-spinner {
      width: 40px;
      height: 40px;
      border: 5px solid #ddd;
      border-top: 5px solid #333;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    /* Mobile specific styles */
    @media (max-width: 768px) {
      .mtg-tooltip {
        max-width: 80%; /* Responsive width */
        width: auto;
        left: 50% !important;
        transform: translateX(-50%);
        bottom: 20px !important;
        top: auto !important;
      }
      .mtg-tooltip-close {
        display: block !important;
      }
      .mtg-tooltip.mobile-active {
        position: fixed;
        z-index: 2000;
      }
      /* Backdrop for mobile */
      .mtg-tooltip-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 1500;
        opacity: 0;
        transition: opacity 0.2s;
        pointer-events: none;
      }
      .mtg-tooltip-backdrop.active {
        opacity: 1;
        pointer-events: auto;
      }
    }
  `;
  document.head.appendChild(style);

  // Create backdrop for mobile
  const backdrop = document.createElement('div');
  backdrop.className = 'mtg-tooltip-backdrop';
  document.body.appendChild(backdrop);

  // Function to fetch card data from Scryfall
  async function fetchCardData(oracleId) {
    if (cardCache.has(oracleId)) {
      return cardCache.get(oracleId);
    }
    
    const url = `https://api.scryfall.com/cards/search?q=oracleid%3A${oracleId}`;
    
    try {
      // Using the browser's default User-Agent as per Scryfall's requirements
      const response = await fetch(url, {
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch card data: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Get the first card's image from the response
      if (data.data && data.data.length > 0) {
        const cardData = {
          name: data.data[0].name,
          image: data.data[0].image_uris?.normal || 
                data.data[0].card_faces?.[0]?.image_uris?.normal || 
                data.data[0].image_uris?.large
        };
        
        // Cache the result
        cardCache.set(oracleId, cardData);
        return cardData;
      } else {
        throw new Error('No card data found');
      }
    } catch (error) {
      console.error('Error fetching card data:', error);
      return null;
    }
  }

  // Function to detect if we're on a mobile device
  function isMobileDevice() {
    return window.innerWidth <= 768 || 
            /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  }

  // Function to show loading state
  function showLoadingState(x, y) {
    tooltip.className = 'mtg-tooltip loading';
    tooltip.innerHTML = '<div class="loading-spinner"></div>';
    tooltip.appendChild(closeButton);
    
    if (isMobileDevice()) {
      // Mobile positioning - centered at bottom
      tooltip.classList.add('mobile-active');
      backdrop.classList.add('active');
    } else {
      // Desktop positioning
      positionTooltip(x, y);
    }
    
    tooltip.style.display = 'flex';
    hoverTimer = setTimeout(() => {
      tooltip.style.opacity = '1';
    }, 10);
  }

  // Function to show card image
  function showCardImage(cardData, x, y) {
    tooltip.className = 'mtg-tooltip';
    tooltip.style.backgroundImage = `url(${cardData.image})`;
    tooltip.style.backgroundSize = 'contain';
    tooltip.style.backgroundRepeat = 'no-repeat';
    tooltip.innerHTML = '';  // Prevents old images from persisting

    tooltip.appendChild(closeButton);
    
    if (isMobileDevice()) {
      // Mobile positioning - centered at bottom
      tooltip.classList.add('mobile-active');
      backdrop.classList.add('active');
      closeButton.style.display = 'block';
    } else {
      // Desktop positioning
      positionTooltip(x, y);
      closeButton.style.display = 'none';
    }
    
    tooltip.style.display = 'block';
    hoverTimer = setTimeout(() => {
      tooltip.style.opacity = '1';
    }, 10);
  }

  // Function to hide tooltip
  function hideTooltip() {
    tooltip.style.opacity = '0';
    backdrop.classList.remove('active');
    activeCardElement = null;
    
    leaveTimer = setTimeout(() => {
      tooltip.style.display = 'none';
      tooltip.classList.remove('mobile-active');
    }, 200);
  }

  // Function to position tooltip for desktop
  function positionTooltip(x, y) {
    const tooltipRect = tooltip.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Adjust position to keep tooltip within viewport
    let posX = x + 20; // 20px offset from cursor
    let posY = y;
    
    // Check if tooltip would go off the right edge
    if (posX + tooltipRect.width > viewportWidth) {
      posX = x - tooltipRect.width - 10;
    }
    
    // Check if tooltip would go off the bottom edge
    if (posY + tooltipRect.height > viewportHeight) {
      posY = viewportHeight - tooltipRect.height - 10;
    }
    
    tooltip.style.left = `${posX}px`;
    tooltip.style.top = `${posY}px`;
  }

  // Handle showing card for both mouse and touch events
  async function showCard(element, x, y) {  
    let oracleId = element.getAttribute('oracleId');
    if (!oracleId) {
      console.error('Missing oracleId attribute on hoverCard element');
      return;
    }

    // If we can't find a card in the lookup for whatever reason, use AWOL (the card).
    if (oracleId == 'None') {
      oracleId = '3bf1ddcd-efde-4888-92f6-718be495b799';
    }
    
    // Toggle behavior for mobile - close if same card is tapped again
    if (isMobileDevice() && element === activeCardElement) {
      hideTooltip();
      return;
    }
    
    activeCardElement = element;
    
    // Get position based on device type
    let posX, posY;
    if (isMobileDevice()) {
      // For mobile, we'll position at bottom center, handled in the show functions
      posX = 0;
      posY = 0;
    } else {
      // For desktop, position near the element
      posX = x;
      posY = y;
    }
    
    tooltip.style.backgroundImage = ''; // Clears old card image
    showLoadingState(posX, posY);
    
    const cardData = await fetchCardData(oracleId);
    if (cardData && element === activeCardElement) {
      showCardImage(cardData, posX, posY);
    } else if (!cardData) {
      hideTooltip();
    }
  }

  // Handle window resize to adjust between mobile/desktop behavior
  let lastIsMobile = isMobileDevice();
  window.addEventListener('resize', function() {
    const nowIsMobile = isMobileDevice();
    if (lastIsMobile !== nowIsMobile) {
      lastIsMobile = nowIsMobile;
      hideTooltip(); // Hide any visible tooltips when switching between modes
    }
  });
})