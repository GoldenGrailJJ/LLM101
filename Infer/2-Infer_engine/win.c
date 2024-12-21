#include "win.h"
#include <errno.h>
#include <io.h>

// Define FILE_MAP_EXECUTE if it is not already defined
#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE    0x0020
#endif /* FILE_MAP_EXECUTE */

// Function to map Windows error codes to errno values
static int __map_mman_error(const uint32_t err, const int deferr)
{
    if (err == 0)
        return 0; // No error
    //TODO: implement error mapping
    return err; // Return the error code
}

// Function to convert protection flags to Windows page protection constants
static uint32_t __map_mmap_prot_page(const int prot)
{
    uint32_t protect = 0; // Initialize protection variable
    
    if (prot == PROT_NONE)
        return protect; // No protection
    
    // Determine the appropriate protection based on the requested flags
    if ((prot & PROT_EXEC) != 0)
    {
        protect = ((prot & PROT_WRITE) != 0) ? 
                    PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    }
    else
    {
        protect = ((prot & PROT_WRITE) != 0) ?
                    PAGE_READWRITE : PAGE_READONLY;
    }
    
    return protect; // Return the calculated protection flags
}

// Function to convert protection flags to desired access for file mapping
static uint32_t __map_mmap_prot_file(const int prot)
{
    uint32_t desiredAccess = 0; // Initialize desired access variable
    
    if (prot == PROT_NONE)
        return desiredAccess; // No access
    
    // Set desired access based on the requested protection flags
    if ((prot & PROT_READ) != 0)
        desiredAccess |= FILE_MAP_READ;
    if ((prot & PROT_WRITE) != 0)
        desiredAccess |= FILE_MAP_WRITE;
    if ((prot & PROT_EXEC) != 0)
        desiredAccess |= FILE_MAP_EXECUTE;
    
    return desiredAccess; // Return the desired access flags
}

// Function to create a memory mapping
void* mmap(void *addr, size_t len, int prot, int flags, int fildes, ssize_t off)
{
    HANDLE fm, h; // File mapping handle and file handle
    void * map = MAP_FAILED; // Initialize map to failed state
    
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4293) // Disable specific warning for integer division
#endif

    // Calculate file offset low and high parts
    const uint32_t dwFileOffsetLow = (uint32_t)(off & 0xFFFFFFFFL);
    const uint32_t dwFileOffsetHigh = (uint32_t)((off >> 32) & 0xFFFFFFFFL);
    const uint32_t protect = __map_mmap_prot_page(prot); // Get page protection
    const uint32_t desiredAccess = __map_mmap_prot_file(prot); // Get desired access

    // Calculate maximum size for mapping
    const ssize_t maxSize = off + (ssize_t)len;
    const uint32_t dwMaxSizeLow = (uint32_t)(maxSize & 0xFFFFFFFFL);
    const uint32_t dwMaxSizeHigh = (uint32_t)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop) // Restore previous warning state
#endif

    errno = 0; // Reset errno
    
    // Validate input parameters
    if (len == 0 
        /* Unsupported flag combinations */
        || (flags & MAP_FIXED) != 0
        /* Unsupported protection combinations */
        || prot == PROT_EXEC)
    {
        errno = EINVAL; // Invalid argument error
        return MAP_FAILED; // Return failure
    }
    
    // Get the file handle, or set to invalid handle for anonymous mapping
    h = ((flags & MAP_ANONYMOUS) == 0) ? 
                    (HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

    // Check for valid file handle if not using anonymous mapping
    if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
    {
        errno = EBADF; // Bad file descriptor error
        return MAP_FAILED; // Return failure
    }

    // Create a file mapping object
    fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

    // Check if file mapping creation failed
    if (fm == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
        return MAP_FAILED; // Return failure
    }

    // Map the view of the file into the address space
    map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

    CloseHandle(fm); // Close the file mapping handle

    // Check if mapping the view failed
    if (map == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
        return MAP_FAILED; // Return failure
    }

    return map; // Return the mapped address
}

// Function to unmap a previously mapped memory region
int munmap(void *addr, size_t len)
{
    if (UnmapViewOfFile(addr)) // Attempt to unmap the view
        return 0; // Success
        
    errno =  __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
    
    return -1; // Return failure
}

// Function to change the protection of a mapped memory region
int mprotect(void *addr, size_t len, int prot)
{
    uint32_t newProtect = __map_mmap_prot_page(prot); // Get new protection flags
    uint32_t oldProtect = 0; // Variable to store old protection flags
    
    // Change the protection of the memory region
    if (VirtualProtect(addr, len, newProtect, &oldProtect))
        return 0; // Success
    
    errno =  __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
    
    return -1; // Return failure
}

// Function to synchronize changes made to a mapped memory region
int msync(void *addr, size_t len, int flags)
{
    if (FlushViewOfFile(addr, len)) // Attempt to flush changes to the file
        return 0; // Success
    
    errno =  __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
    
    return -1; // Return failure
}

// Function to lock a memory region in physical memory
int mlock(const void *addr, size_t len)
{
    if (VirtualLock((LPVOID)addr, len)) // Attempt to lock the memory
        return 0; // Success
        
    errno =  __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
    
    return -1; // Return failure
}

// Function to unlock a previously locked memory region
int munlock(const void *addr, size_t len)
{
    if (VirtualUnlock((LPVOID)addr, len)) // Attempt to unlock the memory
        return 0; // Success
        
    errno =  __map_mman_error(GetLastError(), EPERM); // Map Windows error to errno
    
    return -1; // Return failure
}

// Portable clock_gettime function for Windows
int clock_gettime(int clk_id, struct timespec *tp) {
    uint32_t ticks = GetTickCount(); // Get the number of milliseconds since the system started
    tp->tv_sec = ticks / 1000; // Set seconds
    tp->tv_nsec = (ticks % 1000) * 1000000; // Set nanoseconds
    return 0; // Success
}