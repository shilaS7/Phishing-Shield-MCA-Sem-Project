/**
* Template Name: Bikin - v2.2.1
* Template URL: https://bootstrapmade.com/bikin-free-simple-landing-page-template/
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/
!(function($) {
  "use strict";

  // Preloader
  $(window).on('load', function() {
    if ($('#preloader').length) {
      $('#preloader').delay(100).fadeOut('slow', function() {
        $(this).remove();
      });
    }
  });

  // Smooth scroll for the navigation menu and links with .scrollto classes
  var scrolltoOffset = $('#header').outerHeight() - 1;
  $(document).on('click', '.nav-menu a, .mobile-nav a, .scrollto', function(e) {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      if (target.length) {
        e.preventDefault();

        var scrollto = target.offset().top - scrolltoOffset;

        if ($(this).attr("href") == '#header') {
          scrollto = 0;
        }

        $('html, body').animate({
          scrollTop: scrollto
        }, 1500, 'easeInOutExpo');

        if ($(this).parents('.nav-menu, .mobile-nav').length) {
          $('.nav-menu .active, .mobile-nav .active').removeClass('active');
          $(this).closest('li').addClass('active');
        }

        if ($('body').hasClass('mobile-nav-active')) {
          $('body').removeClass('mobile-nav-active');
          $('.mobile-nav-toggle i').toggleClass('icofont-navigation-menu icofont-close');
          $('.mobile-nav-overly').fadeOut();
        }
        return false;
      }
    }
  });

  // Activate smooth scroll on page load with hash links in the url
  $(document).ready(function() {
    if (window.location.hash) {
      var initial_nav = window.location.hash;
      if ($(initial_nav).length) {
        var scrollto = $(initial_nav).offset().top - scrolltoOffset;
        $('html, body').animate({
          scrollTop: scrollto
        }, 1500, 'easeInOutExpo');
      }
    }
  });

  // Mobile Navigation
  if ($('.nav-menu').length) {
    var $mobile_nav = $('.nav-menu').clone().prop({
      class: 'mobile-nav d-lg-none'
    });
    $('body').append($mobile_nav);
    $('body').prepend('<button type="button" class="mobile-nav-toggle d-lg-none"><i class="icofont-navigation-menu"></i></button>');
    $('body').append('<div class="mobile-nav-overly"></div>');

    $(document).on('click', '.mobile-nav-toggle', function(e) {
      $('body').toggleClass('mobile-nav-active');
      $('.mobile-nav-toggle i').toggleClass('icofont-navigation-menu icofont-close');
      $('.mobile-nav-overly').toggle();
    });

    $(document).on('click', '.mobile-nav .drop-down > a', function(e) {
      e.preventDefault();
      $(this).next().slideToggle(300);
      $(this).parent().toggleClass('active');
    });

    $(document).click(function(e) {
      var container = $(".mobile-nav, .mobile-nav-toggle");
      if (!container.is(e.target) && container.has(e.target).length === 0) {
        if ($('body').hasClass('mobile-nav-active')) {
          $('body').removeClass('mobile-nav-active');
          $('.mobile-nav-toggle i').toggleClass('icofont-navigation-menu icofont-close');
          $('.mobile-nav-overly').fadeOut();
        }
      }
    });
  } else if ($(".mobile-nav, .mobile-nav-toggle").length) {
    $(".mobile-nav, .mobile-nav-toggle").hide();
  }

  // Navigation active state on scroll - disabled for better page-based navigation
  // The active state is now handled by server-side template rendering
  
  // Keep navigation highlighting based on current page instead of scroll position
  $(document).ready(function() {
    // Don't change active states on scroll - let the server-side templates handle it
    // This prevents incorrect highlighting when scrolling on different pages
  });

  // Back to top button
  $(window).scroll(function() {
    if ($(this).scrollTop() > 100) {
      $('.back-to-top').fadeIn('slow');
    } else {
      $('.back-to-top').fadeOut('slow');
    }
  });

  $('.back-to-top').click(function() {
    $('html, body').animate({
      scrollTop: 0
    }, 1500, 'easeInOutExpo');
    return false;
  });

  // Testimonials carousel (uses the Owl Carousel library)
  $(".testimonials-carousel").owlCarousel({
    autoplay: true,
	scroll: false,
    dots: true,
	slideBy: 1,
    responsive: {
      0: {
        items: 1
      },
      768: {
        items: 1
      },
      900: {
        items: 2
      }
    }
  });

  // Porfolio isotope and filter
  $(window).on('load', function() {
    var portfolioIsotope = $('.portfolio-container').isotope({
      itemSelector: '.portfolio-item'
    });

    $('#portfolio-flters li').on('click', function() {
      $("#portfolio-flters li").removeClass('filter-active');
      $(this).addClass('filter-active');

      portfolioIsotope.isotope({
        filter: $(this).data('filter')
      });
      aos_init();
    });

    // Initiate venobox (lightbox feature used in portofilo)
    $(document).ready(function() {
      $('.venobox').venobox();
    });
  });

  // Portfolio details carousel
  $(".portfolio-details-carousel").owlCarousel({
    autoplay: true,
    dots: true,
    loop: true,
    items: 1
  });

  // Init AOS
  function aos_init() {
    AOS.init({
      duration: 1000,
      once: true
    });
  }
  $(window).on('load', function() {
    aos_init();
  });

  // Enhanced Interactive Features
  $(document).ready(function() {
    
    // Add loading spinner to form submission
    $('#hero form').on('submit', function() {
      var submitBtn = $(this).find('button[type="submit"]');
      var originalText = submitBtn.text();
      submitBtn.html('<span class="loading-spinner"></span>Scanning...');
      submitBtn.prop('disabled', true);
      
      // Re-enable button after form processes (you may want to adjust timing)
      setTimeout(function() {
        submitBtn.html(originalText);
        submitBtn.prop('disabled', false);
      }, 3000);
    });

    // Add typing animation to main heading
    function typeWriter(element, text, speed = 100) {
      let i = 0;
      element.innerHTML = '';
      function typing() {
        if (i < text.length) {
          element.innerHTML += text.charAt(i);
          i++;
          setTimeout(typing, speed);
        } else {
          element.classList.add('typing-animation');
        }
      }
      typing();
    }

    // Apply typing animation to main heading
    const mainHeading = document.querySelector('#hero h1');
    if (mainHeading) {
      const originalText = mainHeading.textContent;
      setTimeout(() => {
        typeWriter(mainHeading, originalText, 80);
      }, 500);
    }

    // Add hover effects to cards and sections
    $('.faq-list li').addClass('card-hover');
    
    // Animate elements on scroll
    $(window).scroll(function() {
      var scrollTop = $(this).scrollTop();
      var windowHeight = $(this).height();
      
      $('.fade-in, .slide-in-left, .slide-in-right').each(function() {
        var elementTop = $(this).offset().top;
        if (scrollTop + windowHeight > elementTop + 100) {
          $(this).addClass('animate');
        }
      });
    });

    // Add pulse animation to important buttons
    $('.button1, .button2').hover(function() {
      $(this).addClass('pulse');
    }, function() {
      $(this).removeClass('pulse');
    });

    // Enhanced FAQ interactions
    $('.faq-list a').click(function(e) {
      e.preventDefault();
      var $this = $(this);
      var $collapse = $this.attr('href');
      
      // Add smooth animation
      $($collapse).slideToggle(400, function() {
        // Add bounce effect when opening
        if ($(this).is(':visible')) {
          $(this).addClass('bounce-in');
          setTimeout(() => {
            $(this).removeClass('bounce-in');
          }, 600);
        }
      });
      
      // Toggle classes for icon animation
      $this.toggleClass('collapsed');
      $this.parent().toggleClass('active');
    });

    // Add smooth hover effects to navigation
    $('.nav-menu a').hover(function() {
      $(this).addClass('nav-hover');
    }, function() {
      $(this).removeClass('nav-hover');
    });

    // Interactive form input enhancements
    $('input, textarea').focus(function() {
      $(this).parent().addClass('focused');
    }).blur(function() {
      $(this).parent().removeClass('focused');
    });

    // Add progress bar animation for confidence scores
    $('.progress-bar').each(function() {
      var $this = $(this);
      var width = $this.attr('style').match(/width:\s*(\d+)%/);
      if (width) {
        $this.addClass('progress-bar-animated');
        $this.css('--progress-width', width[1] + '%');
      }
    });

    // Smooth scroll with easing for internal links
    $('a[href^="#"]').click(function(e) {
      e.preventDefault();
      var target = $($(this).attr('href'));
      if (target.length) {
        $('html, body').animate({
          scrollTop: target.offset().top - 100
        }, 800, 'easeInOutCubic');
      }
    });

    // Add ripple effect to buttons
    $('.button1, .button2, button[type="submit"]').click(function(e) {
      var $this = $(this);
      var offset = $this.offset();
      var x = e.pageX - offset.left;
      var y = e.pageY - offset.top;
      
      var ripple = $('<span class="ripple"></span>');
      ripple.css({
        position: 'absolute',
        left: x,
        top: y,
        transform: 'translate(-50%, -50%)',
        width: '0',
        height: '0',
        borderRadius: '50%',
        background: 'rgba(255,255,255,0.5)',
        animation: 'ripple-animation 0.6s ease-out'
      });
      
      $this.append(ripple);
      
      setTimeout(() => {
        ripple.remove();
      }, 600);
    });

    // Add CSS for ripple animation
    $('<style>')
      .prop('type', 'text/css')
      .html(`
        @keyframes ripple-animation {
          0% {
            width: 0;
            height: 0;
            opacity: 1;
          }
          100% {
            width: 200px;
            height: 200px;
            opacity: 0;
          }
        }
        .bounce-in {
          animation: bounceIn 0.6s ease-out;
        }
        .nav-hover {
          transition: all 0.3s ease;
        }
        .focused {
          transform: scale(1.02);
        }
      `)
      .appendTo('head');

    // Parallax effect for hero section
    $(window).scroll(function() {
      var scrolled = $(this).scrollTop();
      var parallax = $('#hero');
      var speed = 0.5;
      
      if (parallax.length) {
        parallax.css('transform', 'translateY(' + (scrolled * speed) + 'px)');
      }
    });

  });

})(jQuery);

