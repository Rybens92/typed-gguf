/* Card t_57cc0179 — the starved-device repro's dummy allocator.
 *
 * Holds N MiB of DEVICE_LOCAL memory on the machine's discrete Vulkan device and sleeps until it
 * is killed, so a `ggufone bench --backend vulkan` child runs against a device that is *provably*
 * as full as the operator's failing run (3272 MiB free of 8192 MiB; the t_dd62ec29 logs). The
 * pressure deliberately does not come from llama.cpp: it comes from outside the process tested.
 *
 *   cc -O2 -o vram_hog vram_hog.c -lvulkan
 *   VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json ./vram_hog 1200      (hold 1200 MiB)
 *
 * Prints `hog: <device> type=<n> held=<n> MiB` once allocated, then sleeps.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define MIB (1024u * 1024u)
#define CHECK(x)                                                              \
    do {                                                                      \
        VkResult _r = (x);                                                    \
        if (_r != VK_SUCCESS) {                                               \
            fprintf(stderr, "hog: %s failed: VkResult %d\n", #x, (int)_r);    \
            return 2;                                                         \
        }                                                                     \
    } while (0)

int main(int argc, char **argv) {
    unsigned long long hold_mib = argc > 1 ? strtoull(argv[1], NULL, 10) : 1024ull;
    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "ggufone-vram-hog";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance instance;
    CHECK(vkCreateInstance(&ici, NULL, &instance));

    uint32_t count = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (count == 0) { fprintf(stderr, "hog: no Vulkan device\n"); return 3; }
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < count; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &props);
        fprintf(stderr, "hog: device %u = %s (type %d)\n", i, props.deviceName,
                (int)props.deviceType);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) chosen = devices[i];
    }
    if (chosen == VK_NULL_HANDLE) chosen = devices[0];
    vkGetPhysicalDeviceProperties(chosen, &props);

    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;   /* no queues needed for raw memory */
    VkDevice device;
    CHECK(vkCreateDevice(chosen, &dci, NULL, &device));

    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(chosen, &memory);
    int type_index = -1;
    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            type_index = (int)i;
            break;
        }
    }
    if (type_index < 0) for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
        if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            type_index = (int)i;
            break;
        }
    if (type_index < 0) { fprintf(stderr, "hog: no device-local memory type\n"); return 4; }

    VkMemoryAllocateInfo alloc = {0};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = (VkDeviceSize)hold_mib * MIB;
    alloc.memoryTypeIndex = (uint32_t)type_index;
    VkDeviceMemory held = VK_NULL_HANDLE;
    VkResult result = vkAllocateMemory(device, &alloc, NULL, &held);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "hog: could not hold %llu MiB (VkResult %d)\n", hold_mib, (int)result);
        return 5;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {0};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    printf("hog: %s held=%llu MiB type=%d\n", props.deviceName, hold_mib, type_index);
    fflush(stdout);
    for (;;) sleep(3600);
    return 0;
}
