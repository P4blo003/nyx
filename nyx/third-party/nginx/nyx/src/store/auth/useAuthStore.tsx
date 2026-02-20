

import { create } from "zustand";
import type { AuthState } from "./types";
import keycloak, { getTokenMinValidity } from "../../libs/keycloak";

let initCalled = false;

export const useAuthStore = create<AuthState>((set) => ({

    // ---- Properties ---- //

    authUser: null,

    isInitializing: true,
    isConnectionError: false,


    // ---- Methods ---- //

    initKeycloak: async() =>
    {
        if (initCalled) return;
        initCalled = true;

        try
        {
            const authenticated = await keycloak.init({ onLoad: "check-sso" });

            if (authenticated)
            {
                const parsed = keycloak.tokenParsed;

                set({
                    authUser: {
                        fullname: parsed?.name ?? "",
                        email: parsed?.email ?? "",
                        token: keycloak.token ?? ""
                    }
                });
            }

            keycloak.onTokenExpired = () =>
            {
                keycloak.updateToken(getTokenMinValidity()).then(() =>
                {
                    set((state) => ({
                        authUser: state.authUser
                            ? { ...state.authUser, token: keycloak.token ?? "" }
                            : null
                    }));
                });
            };
        }
        catch
        {
            set({ isConnectionError: true });
        }
        finally
        {
            set({ isInitializing: false });
        }
    },

    login: () =>
    {
        keycloak.login();
    },

    logout: () =>
    {
        keycloak.logout({redirectUri: window.location.origin});
        set({ authUser: null });
    }
}))
