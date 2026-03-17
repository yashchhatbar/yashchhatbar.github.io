/**
 * Chatling Enhancements
 * Portfolio Chat Improvements
 * 
 * Features:
 * - Reset chat session on refresh
 * - Auto-open chatbot
 * - Rename bot header
 * - Custom welcome message
 * - Safer widget detection
 * - Manual reset support
 */

(function () {

    /* ----------------------------------
       CONFIGURATION
    ---------------------------------- */

    const CONFIG = {
        BOT_NAME: "Yash AI Assistant",
        WELCOME_MESSAGE:
            "Hi! I'm Yash AI Assistant 👋\n\nI can help you explore Yash's projects, skills, and experience.\n\nTry asking:\n• What projects has Yash built?\n• What are Yash's AI skills?\n• How can I contact Yash?\n• Show Yash's GitHub",
        AUTO_OPEN_DELAY: 1500
    };

    /* ----------------------------------
       RESET CHAT SESSION
    ---------------------------------- */

    function resetChat() {

        try {

            Object.keys(localStorage).forEach(function (key) {

                if (key.startsWith("chtl")) {
                    localStorage.removeItem(key);
                }

            });

        } catch (e) {

            console.warn("Chat reset failed:", e);

        }

    }

    /* Run BEFORE widget loads */
    resetChat();


    /* ----------------------------------
       FIND CHATLING WIDGET
    ---------------------------------- */

    function findWidget() {

        return document.querySelector('[id^="chtl-widget"]');

    }


    /* ----------------------------------
       AUTO OPEN CHATBOT
    ---------------------------------- */

    function autoOpenChat() {

        setTimeout(function () {

            const widget = findWidget();

            if (widget && !sessionStorage.getItem("chatAutoOpened")) {

                widget.click();

                sessionStorage.setItem("chatAutoOpened", "true");

                console.log("Chatbot auto-opened");

            }

        }, CONFIG.AUTO_OPEN_DELAY);

    }


    /* ----------------------------------
       CUSTOMIZE BOT HEADER
    ---------------------------------- */

    function renameBotHeader() {

        try {

            const header = document.querySelector('[class*="chat-header"] h3');

            if (header) {

                header.textContent = CONFIG.BOT_NAME;

            }

        } catch (e) {

            console.warn("Bot rename failed:", e);

        }

    }


    /* ----------------------------------
       ADD CUSTOM WELCOME MESSAGE
    ---------------------------------- */

    function injectWelcomeMessage() {

        try {

            const chatBody = document.querySelector('[class*="chat-body"]');

            if (!chatBody) return;

            const existingMessage = chatBody.querySelector(".custom-welcome");

            if (existingMessage) return;

            const messageHTML = `
                <div class="custom-welcome" style="
                    padding:12px 14px;
                    margin:10px;
                    border-radius:10px;
                    background:#f4f6f8;
                    font-size:14px;
                    line-height:1.5;
                    color:#333;
                ">
                    ${CONFIG.WELCOME_MESSAGE.replace(/\n/g, "<br>")}
                </div>
            `;

            chatBody.insertAdjacentHTML("afterbegin", messageHTML);

        } catch (e) {

            console.warn("Welcome message injection failed:", e);

        }

    }


    /* ----------------------------------
       WIDGET OBSERVER
       Detect when Chatling loads
    ---------------------------------- */

    function observeWidget() {

        const observer = new MutationObserver(function () {

            const widget = findWidget();

            if (widget) {

                renameBotHeader();
                injectWelcomeMessage();

            }

        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

    }


    /* ----------------------------------
       MANUAL RESET FUNCTION
    ---------------------------------- */

    window.startNewChat = function () {

        try {

            Object.keys(localStorage).forEach(function (key) {

                if (key.startsWith("chtl")) {
                    localStorage.removeItem(key);
                }

            });

            sessionStorage.removeItem("chatAutoOpened");

            location.reload();

        } catch (e) {

            console.warn("Manual reset failed:", e);

        }

    };


    /* ----------------------------------
       INIT
    ---------------------------------- */

    window.addEventListener("load", function () {

        autoOpenChat();
        observeWidget();

    });

})();