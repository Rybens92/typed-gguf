/* A minimal, low-perturbation SIGSEGV backtrace for card t_97f1bc93.
 *
 * The crash under study happens at process teardown, where gdb either changes the timing or
 * reports nothing useful ("No stack"); `core_pattern` is read-only in this container, so a core
 * is not an option either. This LD_PRELOAD handler prints the fault address plus a libc
 * backtrace to stderr and then exits with the shell's signal code (128 + 11 = 139), so a caller
 * still sees the crash's own exit status.
 *
 * Build: cc -shared -fPIC -O2 -o segv_bt.so segv_bt.c
 * Use:   LD_PRELOAD=/path/segv_bt.so <command>
 */
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void segv_handler(int sig, siginfo_t *info, void *ctx) {
    (void) ctx;
    char header[256];
    int n = snprintf(header, sizeof header,
                     "\n######## SEGV_BT: signal %d (%s) fault_address=%p ########\n",
                     sig, strsignal(sig), info ? info->si_addr : NULL);
    if (n > 0) {
        ssize_t written = write(2, header, (size_t) n);
        (void) written;
    }
    void *frames[80];
    int count = backtrace(frames, 80);
    backtrace_symbols_fd(frames, count, 2);
    static const char foot[] = "######## SEGV_BT END ########\n";
    ssize_t written = write(2, foot, sizeof foot - 1);
    (void) written;
    _exit(139);
}

__attribute__((constructor)) static void segv_bt_install(void) {
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_sigaction = segv_handler;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGSEGV, &action, NULL);
}
